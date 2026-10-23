# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Co-evolution PPO trainer with native GRPO and mutual on-policy distillation.
"""

import json
import os
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional
import wandb as _wandb
import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt  # find_latest_ckpt replaced by _find_latest_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, get_kl_controller
from .dual_ray_trainer import _prefix_metrics, compute_advantage
from .metrics import compute_data_metrics, compute_length_metrics, reduce_metrics
from .ray_trainer import apply_kl_penalty


@dataclass
class BranchState:
    name: str
    config: PPOConfig
    actor_rollout_ref_wg: RayWorkerGroup
    critic_wg: Optional[RayWorkerGroup]
    kl_ctrl: Any
    reward_fn: Any = None
    val_reward_fn: Any = None


class CoEvolRayPPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        text_train_dataloader: StatefulDataLoader,
        text_val_dataloader: StatefulDataLoader,
        vision_train_dataloader: StatefulDataLoader,
        vision_val_dataloader: StatefulDataLoader,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
        reward_fn_text: Optional[AutoRewardManager] = None,
        reward_fn_vision: Optional[AutoRewardManager] = None,
        val_reward_fn_text: Optional[AutoRewardManager] = None,
        val_reward_fn_vision: Optional[AutoRewardManager] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.text_train_dataloader = text_train_dataloader
        self.text_val_dataloader = text_val_dataloader
        self.vision_train_dataloader = vision_train_dataloader
        self.vision_val_dataloader = vision_val_dataloader
        self.ray_worker_group_cls = ray_worker_group_cls
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        # Per-branch reward fns; fall back to shared reward_fn if not provided
        self._reward_fn_text = reward_fn_text or reward_fn
        self._reward_fn_vision = reward_fn_vision or reward_fn
        self._val_reward_fn_text = val_reward_fn_text or val_reward_fn
        self._val_reward_fn_vision = val_reward_fn_vision or val_reward_fn

        if not self.config.coev.enabled:
            raise ValueError("CoEvolRayPPOTrainer requires `coev.enabled=true`.")
        if not self.config.parallel.enabled:
            raise ValueError("CoEvolRayPPOTrainer requires `parallel.enabled=true`.")
        if self.config.coev.mode != "cross_reward":
            raise NotImplementedError(
                f"Unsupported co-evolution mode: {self.config.coev.mode}. Use 'cross_reward'."
            )
        if self.config.coev.cross_reward_style not in ("reward_bonus", "reverse_kl", "reverse_kl_positive"):
            raise ValueError(
                f"Unsupported cross_reward_style: {self.config.coev.cross_reward_style}. "
                "Use 'reward_bonus', 'reverse_kl', or 'reverse_kl_positive'."
            )
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("CoEvolRayPPOTrainer does not support REMAX yet.")

        self.use_critic = self.config.algorithm.adv_estimator == AdvantageEstimator.GAE
        if self.config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {self.config.algorithm.adv_estimator}.")
        if (
            self.config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and self.config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if self.config.trainer.max_steps is not None:
            self.training_steps = self.config.trainer.max_steps
        elif self.config.data.mini_rollout_batch_size is not None:
            num_examples = max(len(text_train_dataloader), len(vision_train_dataloader)) * self.config.data.mini_rollout_batch_size
            self.training_steps = num_examples // self.config.data.rollout_batch_size * self.config.trainer.total_epochs
        else:
            self.training_steps = max(len(text_train_dataloader), len(vision_train_dataloader)) * self.config.trainer.total_epochs

        self.node_allocation = self._resolve_node_allocation()
        self.num_gpus_per_model = self.config.parallel.n_gpus_per_model
        if self.node_allocation is None:
            if self.num_gpus_per_model is None:
                self.num_gpus_per_model = self.config.trainer.n_gpus_per_node // 2
            if self.num_gpus_per_model <= 0:
                raise ValueError("`parallel.n_gpus_per_model` must be positive.")
            if self.num_gpus_per_model * 2 > self.config.trainer.n_gpus_per_node:
                raise ValueError("Two parallel models require `2 * n_gpus_per_model <= trainer.n_gpus_per_node`.")

        self.branch_config_text = deepcopy(self.config)
        self.branch_config_vision = deepcopy(self.config)
        self._apply_branch_overrides()
        self._prepare_branch_optim_steps(self.branch_config_text)
        self._prepare_branch_optim_steps(self.branch_config_vision)

        self.text_step = 0
        self.vision_step = 0
        self.text_val_reward_score = 0.0
        self.vision_val_reward_score = 0.0
        self.best_text_val_reward_score = -1.0
        self.best_vision_val_reward_score = -1.0
        self.best_text_step = None
        self.best_vision_step = None
        self._tracker_lock = threading.Lock()
        self._logger_lock = threading.Lock()
        # Adaptive coef: multiplier on reverse_kl_coef per direction (1.0 = normal, 0.0 = disabled)
        self._adaptive_coef_text = 1.0   # text as teacher → vision as student
        self._adaptive_coef_vision = 1.0  # vision as teacher → text as student
        self.wg_dicts = []




    def _resolve_node_allocation(self) -> Optional[dict[str, list[int]]]:
        """Build full-node resource pools when both dual-branch node counts are set."""
        nodes_text = self.config.parallel.nodes_text
        nodes_vision = self.config.parallel.nodes_vision
        if nodes_text is None and nodes_vision is None:
            return None
        if nodes_text is None or nodes_vision is None:
            raise ValueError(
                "Dual-branch machine-level allocation requires both "
                "`parallel.nodes_text` and `parallel.nodes_vision` to be set."
            )
        if nodes_text <= 0 or nodes_vision <= 0:
            raise ValueError(
                "Dual-branch node counts must be positive, got "
                f"nodes_text={nodes_text}, nodes_vision={nodes_vision}."
            )

        nnodes = self.config.trainer.nnodes
        total_nodes = nodes_text + nodes_vision
        if total_nodes > nnodes:
            raise ValueError(
                f"Machine-level allocation needs {total_nodes} nodes "
                f"(text={nodes_text}, vision={nodes_vision}), but trainer.nnodes={nnodes}."
            )
        if total_nodes < nnodes:
            print(
                f"[dual-branch] Warning: {nnodes - total_nodes} node(s) will be idle. "
                f"Allocated {total_nodes} out of {nnodes}.",
                flush=True,
            )

        gpus_per_node = self.config.trainer.n_gpus_per_node
        if gpus_per_node <= 0:
            raise ValueError(f"`trainer.n_gpus_per_node` must be positive, got {gpus_per_node}.")

        allocation = {
            "text": [gpus_per_node] * nodes_text,
            "vision": [gpus_per_node] * nodes_vision,
        }
        print(
            "[dual-branch] Machine-level node allocation: "
            f"text={nodes_text} node(s), vision={nodes_vision} node(s), "
            f"{gpus_per_node} GPUs per node.",
            flush=True,
        )
        return allocation

    def _apply_branch_overrides(self):
        seed_text = self.config.parallel.rollout_seed_a
        seed_vision = self.config.parallel.rollout_seed_b
        if seed_text is None:
            seed_text = self.config.worker.rollout.seed
        if seed_vision is None:
            seed_vision = self.config.worker.rollout.seed + 1

        self.branch_config_text.worker.rollout.seed = seed_text
        self.branch_config_vision.worker.rollout.seed = seed_vision

        if self.config.parallel.rollout_temperature_a is not None:
            self.branch_config_text.worker.rollout.temperature = self.config.parallel.rollout_temperature_a
        if self.config.parallel.rollout_temperature_b is not None:
            self.branch_config_vision.worker.rollout.temperature = self.config.parallel.rollout_temperature_b
        if self.config.parallel.rollout_top_p_a is not None:
            self.branch_config_text.worker.rollout.top_p = self.config.parallel.rollout_top_p_a
        if self.config.parallel.rollout_top_p_b is not None:
            self.branch_config_vision.worker.rollout.top_p = self.config.parallel.rollout_top_p_b

        if self.config.parallel.rollout_batch_size_a is not None:
            self.branch_config_text.data.rollout_batch_size = self.config.parallel.rollout_batch_size_a
        if self.config.parallel.rollout_batch_size_b is not None:
            self.branch_config_vision.data.rollout_batch_size = self.config.parallel.rollout_batch_size_b

        if self.config.parallel.actor_global_batch_size_a is not None:
            self.branch_config_text.worker.actor.global_batch_size = self.config.parallel.actor_global_batch_size_a
        if self.config.parallel.actor_global_batch_size_b is not None:
            self.branch_config_vision.worker.actor.global_batch_size = self.config.parallel.actor_global_batch_size_b
        if self.config.parallel.critic_global_batch_size_a is not None:
            self.branch_config_text.worker.critic.global_batch_size = self.config.parallel.critic_global_batch_size_a
        if self.config.parallel.critic_global_batch_size_b is not None:
            self.branch_config_vision.worker.critic.global_batch_size = self.config.parallel.critic_global_batch_size_b

        # Per-branch model path overrides (works with any mode)
        if self.config.parallel.model_path_a is not None:
            self.branch_config_text.worker.actor.model.model_path = self.config.parallel.model_path_a
        if self.config.parallel.model_path_b is not None:
            self.branch_config_vision.worker.actor.model.model_path = self.config.parallel.model_path_b

        self.branch_config_text.worker.actor.use_kl_loss = False
        self.branch_config_vision.worker.actor.use_kl_loss = False
        self.branch_config_vision.worker.actor.kl_coef = self.config.coev.anchor_coef
        self.branch_config_vision.worker.actor.kl_penalty = self.config.algorithm.kl_penalty

    def _prepare_branch_optim_steps(self, branch_config: PPOConfig):
        branch_config.worker.actor.optim.training_steps = self.training_steps
        branch_config.worker.critic.optim.training_steps = self.training_steps

    def init_workers(self):
        if self.node_allocation is None:
            process_on_nodes_text = [self.num_gpus_per_model] * self.config.trainer.nnodes
            process_on_nodes_vision = [self.num_gpus_per_model] * self.config.trainer.nnodes
        else:
            process_on_nodes_text = self.node_allocation["text"]
            process_on_nodes_vision = self.node_allocation["vision"]

        pool_text = RayResourcePool(
            process_on_nodes=process_on_nodes_text,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="coev_text",
        )
        pool_vision = RayResourcePool(
            process_on_nodes=process_on_nodes_vision,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="coev_vision",
        )
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = sum(process_on_nodes_text) + sum(process_on_nodes_vision)
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than required GPUs {gpus_required}.")

        # Initialize branches sequentially — parallel init causes NCCL process group
        # rendezvous conflicts (both branches race for the same TCP port).
        print("[init_workers] Initializing branch_text...", flush=True)
        self.branch_text = self._init_branch("branch_text", pool_text, self.branch_config_text)
        print("[init_workers] branch_text done. Initializing branch_vision...", flush=True)
        self.branch_vision = self._init_branch("branch_vision", pool_vision, self.branch_config_vision)
        print("[init_workers] branch_vision done.", flush=True)

        self.branch_text.reward_fn = self._reward_fn_text
        self.branch_text.val_reward_fn = self._val_reward_fn_text
        self.branch_vision.reward_fn = self._reward_fn_vision
        self.branch_vision.val_reward_fn = self._val_reward_fn_vision

    def _init_branch(self, branch_name: str, resource_pool: RayResourcePool, branch_config: PPOConfig) -> BranchState:
        t0 = time.time()
        print(f"  [{branch_name}] Creating class_dict...", flush=True)
        class_dict = {
            "actor_rollout_ref": RayClassWithInitArgs(
                cls=ray.remote(FSDPWorker),
                config=branch_config.worker,
                role="actor_rollout_ref",
            )
        }
        if self.use_critic:
            class_dict["critic"] = RayClassWithInitArgs(
                cls=ray.remote(FSDPWorker),
                config=branch_config.worker,
                role="critic",
            )

        print(f"  [{branch_name}] Creating worker_dict_cls...", flush=True)
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        print(f"  [{branch_name}] Creating worker group...", flush=True)
        wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        print(f"  [{branch_name}] Spawning workers...", flush=True)
        spawned = wg_dict.spawn(prefix_set=class_dict.keys())
        print(f"  [{branch_name}] Workers spawned in {time.time() - t0:.1f}s", flush=True)
        self.wg_dicts.append(wg_dict)

        critic_wg = spawned.get("critic")
        if critic_wg is not None:
            print(f"  [{branch_name}] Calling critic.init_model()...", flush=True)
            critic_wg.init_model()
            print(f"  [{branch_name}] critic.init_model() done in {time.time() - t0:.1f}s", flush=True)
        actor_rollout_ref_wg = spawned["actor_rollout_ref"]
        print(f"  [{branch_name}] Calling actor_rollout_ref.init_model()...", flush=True)
        actor_rollout_ref_wg.init_model()
        print(f"  [{branch_name}] actor_rollout_ref.init_model() done in {time.time() - t0:.1f}s", flush=True)

        kl_ctrl = (
            get_kl_controller(branch_config.algorithm)
            if not branch_config.algorithm.disable_kl
            else FixedKLController(init_kl_coef=0.0)
        )
        return BranchState(
            name=branch_name,
            config=branch_config,
            actor_rollout_ref_wg=actor_rollout_ref_wg,
            critic_wg=critic_wg,
            kl_ctrl=kl_ctrl,
        )

    def _save_branch_checkpoint(self, branch: BranchState, step: int, dataloader: StatefulDataLoader, dataloader_name: str):
        """Save a single branch checkpoint independently."""
        branch_short = branch.name.replace("branch_", "")  # "text" or "vision"
        dir_format = f"{branch.name}_step_{{}}"

        # Clean up old per-branch checkpoints
        is_text = branch.name == "branch_text"
        best_step_attr = "best_text_step" if is_text else "best_vision_step"
        best_score_attr = "best_text_val_reward_score" if is_text else "best_vision_val_reward_score"
        score_attr = "text_val_reward_score" if is_text else "vision_val_reward_score"

        if getattr(self, score_attr) > getattr(self, best_score_attr):
            setattr(self, best_score_attr, getattr(self, score_attr))
            setattr(self, best_step_attr, step)

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            step,
            getattr(self, best_step_attr) or 0,
            self.config.trainer.save_limit,
            directory_format=dir_format,
        )

        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, dir_format.format(step))
        actor_path = os.path.join(folder_path, f"actor_{branch_short}")
        branch.actor_rollout_ref_wg.save_checkpoint(
            actor_path, save_model_only=self.config.trainer.save_model_only
        )
        if self.use_critic and branch.critic_wg is not None:
            critic_path = os.path.join(folder_path, f"critic_{branch_short}")
            branch.critic_wg.save_checkpoint(
                critic_path, save_model_only=self.config.trainer.save_model_only
            )
        torch.save(dataloader.state_dict(), os.path.join(folder_path, f"{dataloader_name}_dataloader.pt"))

        # Update tracker under lock to avoid race between threads
        with self._tracker_lock:
            tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
            existing = {}
            if os.path.exists(tracker_path):
                with open(tracker_path, "r") as f:
                    existing = json.load(f)
            existing.update({
                "best_text_step": self.best_text_step,
                "best_text_val_reward_score": round(self.best_text_val_reward_score, 4),
                "best_vision_step": self.best_vision_step,
                "best_vision_val_reward_score": round(self.best_vision_val_reward_score, 4),
                "last_text_step": self.text_step,
                "last_vision_step": self.vision_step,
            })
            if is_text:
                existing["last_text_ckpt_dir"] = dir_format.format(step)
            else:
                existing["last_vision_ckpt_dir"] = dir_format.format(step)
            with open(tracker_path, "w") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

    def _save_checkpoint(self):
        """Save both branches together (for final/coordinated saves)."""
        folder_path = os.path.join(
            self.config.trainer.save_checkpoint_path,
            f"step_t{self.text_step}_v{self.vision_step}",
        )
        actor_text_path = os.path.join(folder_path, "actor_text")
        actor_vision_path = os.path.join(folder_path, "actor_vision")
        self.branch_text.actor_rollout_ref_wg.save_checkpoint(
            actor_text_path, save_model_only=self.config.trainer.save_model_only
        )
        self.branch_vision.actor_rollout_ref_wg.save_checkpoint(
            actor_vision_path, save_model_only=self.config.trainer.save_model_only
        )

        if self.use_critic:
            critic_text_path = os.path.join(folder_path, "critic_text")
            critic_vision_path = os.path.join(folder_path, "critic_vision")
            self.branch_text.critic_wg.save_checkpoint(
                critic_text_path, save_model_only=self.config.trainer.save_model_only
            )
            self.branch_vision.critic_wg.save_checkpoint(
                critic_vision_path, save_model_only=self.config.trainer.save_model_only
            )

        torch.save(self.text_train_dataloader.state_dict(), os.path.join(folder_path, "text_dataloader.pt"))
        torch.save(self.vision_train_dataloader.state_dict(), os.path.join(folder_path, "vision_dataloader.pt"))

        tracker_info = {
            "best_text_step": self.best_text_step,
            "best_text_val_reward_score": round(self.best_text_val_reward_score, 4),
            "best_vision_step": self.best_vision_step,
            "best_vision_val_reward_score": round(self.best_vision_val_reward_score, 4),
            "last_text_step": self.text_step,
            "last_vision_step": self.vision_step,
            "last_actor_text_path": os.path.abspath(actor_text_path),
            "last_actor_vision_path": os.path.abspath(actor_vision_path),
        }
        tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(tracker_path, "w") as f:
            json.dump(tracker_info, f, ensure_ascii=False, indent=2)

    def _find_latest_ckpt(self) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        """Find the latest combined checkpoint using our tracker format.

        Supports both new format (last_text_step/last_vision_step) and legacy
        format (last_global_step) written by the old single-loop trainer.
        """
        save_path = self.config.trainer.save_checkpoint_path
        if save_path is None:
            return None, None
        tracker_path = os.path.join(save_path, CHECKPOINT_TRACKER)
        if not os.path.exists(tracker_path):
            return None, None

        with open(tracker_path, "r") as f:
            tracker_info = json.load(f)

        # New format: step_t{text}_v{vision}
        if "last_text_step" in tracker_info and "last_vision_step" in tracker_info:
            t = tracker_info["last_text_step"]
            v = tracker_info["last_vision_step"]
            ckpt_path = os.path.join(save_path, f"step_t{t}_v{v}")
            if os.path.exists(ckpt_path):
                print(f"Found latest checkpoint: {ckpt_path}, will resume from it.")
                return ckpt_path, tracker_info

        # Per-branch format: branch_text_step_{N} + branch_vision_step_{M}
        text_ckpt_dir = tracker_info.get("last_text_ckpt_dir")
        vision_ckpt_dir = tracker_info.get("last_vision_ckpt_dir")
        if text_ckpt_dir and vision_ckpt_dir:
            text_ckpt_path = os.path.join(save_path, text_ckpt_dir)
            vision_ckpt_path = os.path.join(save_path, vision_ckpt_dir)
            if os.path.exists(text_ckpt_path) and os.path.exists(vision_ckpt_path):
                print(f"Found per-branch checkpoints: text={text_ckpt_path}, vision={vision_ckpt_path}, will resume.")
                return "__per_branch__", tracker_info

        # Legacy format: global_step_{N}
        if "last_global_step" in tracker_info:
            step = tracker_info["last_global_step"]
            ckpt_path = os.path.join(save_path, f"global_step_{step}")
            if os.path.exists(ckpt_path):
                print(f"Found latest legacy checkpoint: {ckpt_path}, will resume from it.")
                return ckpt_path, tracker_info

        print("Checkpoint tracker found but no matching checkpoint directory exists.")
        return None, None

    def _load_checkpoint(self):
        tracker_info = None
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = self._find_latest_ckpt()
            if tracker_info is not None:
                self.best_text_val_reward_score = tracker_info.get("best_text_val_reward_score", 0.0)
                self.best_vision_val_reward_score = tracker_info.get("best_vision_val_reward_score", 0.0)
                self.best_text_step = tracker_info.get("best_text_step", 0)
                self.best_vision_step = tracker_info.get("best_vision_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        # Per-branch checkpoint restore
        if load_checkpoint_path == "__per_branch__":
            self._load_per_branch_checkpoint(tracker_info)
            return

        folder_name = load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]

        # Support new format: step_t{text}_v{vision}
        if folder_name.startswith("step_t"):
            parts = folder_name.replace("step_t", "").split("_v")
            self.text_step = int(parts[0])
            self.vision_step = int(parts[1])
        # Support legacy format: global_step_{N}
        elif "global_step_" in folder_name:
            global_step = int(folder_name.split("global_step_")[-1])
            self.text_step = global_step
            self.vision_step = global_step
        else:
            raise ValueError(
                f"`load_checkpoint_path` should end with `step_t<N>_v<N>` or `global_step_<N>`, got: {folder_name}"
            )

        print(f"Load from checkpoint: {load_checkpoint_path} (text_step={self.text_step}, vision_step={self.vision_step}).")

        self.branch_text.actor_rollout_ref_wg.load_checkpoint(os.path.join(load_checkpoint_path, "actor_text"))
        self.branch_vision.actor_rollout_ref_wg.load_checkpoint(os.path.join(load_checkpoint_path, "actor_vision"))
        if self.use_critic:
            self.branch_text.critic_wg.load_checkpoint(os.path.join(load_checkpoint_path, "critic_text"))
            self.branch_vision.critic_wg.load_checkpoint(os.path.join(load_checkpoint_path, "critic_vision"))

        text_dataloader_path = os.path.join(load_checkpoint_path, "text_dataloader.pt")
        if os.path.exists(text_dataloader_path):
            self.text_train_dataloader.load_state_dict(torch.load(text_dataloader_path, weights_only=False))
        vision_dataloader_path = os.path.join(load_checkpoint_path, "vision_dataloader.pt")
        if os.path.exists(vision_dataloader_path):
            self.vision_train_dataloader.load_state_dict(torch.load(vision_dataloader_path, weights_only=False))

    def _load_per_branch_checkpoint(self, tracker_info: dict[str, Any]):
        """Load from separate per-branch checkpoint directories."""
        save_path = self.config.trainer.save_checkpoint_path
        text_dir = tracker_info["last_text_ckpt_dir"]
        vision_dir = tracker_info["last_vision_ckpt_dir"]
        text_ckpt_path = os.path.join(save_path, text_dir)
        vision_ckpt_path = os.path.join(save_path, vision_dir)

        self.text_step = int(text_dir.split("_step_")[-1])
        self.vision_step = int(vision_dir.split("_step_")[-1])

        print(
            f"Load from per-branch checkpoints: text={text_ckpt_path} (step={self.text_step}), "
            f"vision={vision_ckpt_path} (step={self.vision_step})."
        )

        self.branch_text.actor_rollout_ref_wg.load_checkpoint(os.path.join(text_ckpt_path, "actor_text"))
        self.branch_vision.actor_rollout_ref_wg.load_checkpoint(os.path.join(vision_ckpt_path, "actor_vision"))
        if self.use_critic:
            self.branch_text.critic_wg.load_checkpoint(os.path.join(text_ckpt_path, "critic_text"))
            self.branch_vision.critic_wg.load_checkpoint(os.path.join(vision_ckpt_path, "critic_vision"))

        # Each per-branch ckpt dir saves {data_name}_dataloader.pt (not branch name).
        # Scan both dirs and load by data name into the correct dataloader.
        for ckpt_path in (text_ckpt_path, vision_ckpt_path):
            text_dl_path = os.path.join(ckpt_path, "text_dataloader.pt")
            if os.path.exists(text_dl_path):
                self.text_train_dataloader.load_state_dict(torch.load(text_dl_path, weights_only=False))
            vision_dl_path = os.path.join(ckpt_path, "vision_dataloader.pt")
            if os.path.exists(vision_dl_path):
                self.vision_train_dataloader.load_state_dict(torch.load(vision_dl_path, weights_only=False))

    def _build_meta_info(self):
        return {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }

    def _make_generation_batch(self, batch: DataProto) -> DataProto:
        return batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "multi_modal_data",
                "offline_outputs",
                "has_offline_trajectory",
                "offline_output",
            ],
            meta_info_keys=[
                "image_min_pixels",
                "image_max_pixels",
                "video_min_pixels",
                "video_max_pixels",
                "video_fps",
                "video_max_frames",
            ],
        )

    def _apply_online_filtering(self, batch: DataProto, metrics_store: defaultdict[str, list], reward_fn) -> DataProto:
        filter_reward_batch = batch.select(
            batch_keys=["responses", "response_mask"],
            non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
        )
        reward_tensor, reward_metrics = ray.get(reward_fn.compute_reward.remote(filter_reward_batch))
        batch.batch["token_level_scores"] = reward_tensor
        for key, value in reward_metrics.items():
            metrics_store[key].extend(value)

        filter_scores = reward_metrics[self.config.algorithm.filter_key]
        uids = batch.non_tensor_batch["uid"]
        uid2scores = defaultdict(list)
        for uid, score in zip(uids, filter_scores):
            uid2scores[uid].append(score)

        uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
        kept_uids = [
            uid
            for uid, avg_score in uid2mean.items()
            if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
        ]
        kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
        if len(kept_sample_idxs) == 0:
            raise RuntimeError("No sample is kept after filtering. Please check your data.")
        return batch[kept_sample_idxs]

    def _next_batch_dict(self, iterator_attr: str, dataloader: StatefulDataLoader):
        iterator = getattr(self, iterator_attr, None)
        if iterator is None:
            iterator = iter(dataloader)
            setattr(self, iterator_attr, iterator)
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            setattr(self, iterator_attr, iterator)
            return next(iterator)


    def _balance_batch(self, batch: DataProto, world_size: int, metrics: dict[str, Any], logging_prefix: str):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        metrics.update(
            _prefix_metrics(
                logging_prefix,
                log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix="global_seqlen"),
            )
        )

    @staticmethod
    def _align_non_tensor_keys(*batches: DataProto) -> None:
        """Keep only the intersection of non_tensor_batch keys across all batches (in-place).

        Different datasets may carry different metadata fields (e.g. text data has
        "success" but vision data doesn't). Aligning prevents DataProto.concat from
        producing mismatched lengths.
        """
        common_keys = set(batches[0].non_tensor_batch.keys())
        for b in batches[1:]:
            common_keys &= set(b.non_tensor_batch.keys())
        for b in batches:
            for key in list(b.non_tensor_batch.keys()):
                if key not in common_keys:
                    del b.non_tensor_batch[key]







    def _compute_cross_reward(
        self,
        scorer_branch: BranchState,
        target_batch: DataProto,
        target_name: str,
        metrics: dict[str, Any],
    ) -> None:
        """Use *scorer_branch* to score *target_batch*'s CoT via log-prob.

        Supports three styles controlled by ``coev.cross_reward_style``:
        - ``reward_bonus``: add ``anchor_coef * mean_log_prob`` to ``token_level_scores``.
        - ``reverse_kl`` / ``reverse_kl_positive``: store ``peer_log_probs`` into batch for deferred advantage mixing.
        """
        cross_batch = target_batch.select(
            batch_keys=["input_ids", "attention_mask", "position_ids", "responses", "response_mask"],
            non_tensor_batch_keys=list(target_batch.non_tensor_batch.keys()),
        )
        cross_log_probs = scorer_branch.actor_rollout_ref_wg.compute_log_probs(cross_batch)
        log_p = cross_log_probs.batch["old_log_probs"]
        mask = target_batch.batch["response_mask"]
        seq_scores = (log_p * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        style = self.config.coev.cross_reward_style
        if style == "reward_bonus":
            anchor_coef = self.config.coev.anchor_coef
            bonus = anchor_coef * seq_scores.unsqueeze(-1) * mask
            target_batch.batch["token_level_scores"] = target_batch.batch["token_level_scores"] + bonus
        elif style in ("reverse_kl", "reverse_kl_positive"):
            # Store peer log-probs for deferred advantage mixing in _optimize_branch
            target_batch.batch["peer_log_probs"] = log_p

        metrics[f"{target_name}/coev/cross_reward_applied"] = 1.0
        metrics[f"{target_name}/coev/cross_reward_mean"] = seq_scores.mean().item()
        metrics[f"{target_name}/coev/cross_reward_std"] = seq_scores.std().item() if seq_scores.numel() > 1 else 0.0

    def _is_opd_active(self, step: int) -> bool:
        """Return whether OPD (cross-reward) should be active at *step*.

        When ``opd_off_steps`` and ``opd_on_steps`` are both 0 (default), fall
        back to the static ``cross_reward_loss_enabled`` flag.  Otherwise cycle:
        first ``opd_off_steps`` steps are pure GRPO, next ``opd_on_steps`` are
        GRPO+OPD, and so on.
        """
        off = self.config.coev.opd_off_steps
        on = self.config.coev.opd_on_steps
        if off <= 0 and on <= 0:
            return self.config.coev.cross_reward_loss_enabled
        cycle = off + on
        if cycle <= 0:
            return self.config.coev.cross_reward_loss_enabled
        return ((step - 1) % cycle) >= off

    def _is_final_opd(self, step: int) -> bool:
        """Return whether final-OPD mode is manually forced."""
        return self.config.coev.force_final_opd

    def _get_reverse_kl_coef(self, branch: BranchState) -> float:
        if branch.name == "branch_text" and self.config.coev.reverse_kl_coef_text is not None:
            base = self.config.coev.reverse_kl_coef_text
        elif branch.name == "branch_vision" and self.config.coev.reverse_kl_coef_vision is not None:
            base = self.config.coev.reverse_kl_coef_vision
        else:
            base = self.config.coev.reverse_kl_coef
        # Adaptive: when branch is student, the teacher is the peer.
        # _adaptive_coef_vision = multiplier when vision is teacher (text is student)
        # _adaptive_coef_text = multiplier when text is teacher (vision is student)
        if self.config.coev.opd_adaptive_coef:
            if branch.name == "branch_text":
                base *= self._adaptive_coef_vision
            else:
                base *= self._adaptive_coef_text
        return base

    def _is_branch_frozen_for_cross_reward(self, branch: BranchState) -> bool:
        if self.config.coev.mode != "cross_reward":
            return False
        if not self.config.coev.freeze_zero_opd_branches:
            return False
        return float(self._get_reverse_kl_coef(branch)) == 0.0

    def _is_branch_pure_opd_for_cross_reward(self, branch: BranchState) -> bool:
        if self.config.coev.mode != "cross_reward":
            return False
        if branch.name == "branch_text":
            return self.config.coev.pure_opd_text
        if branch.name == "branch_vision":
            return self.config.coev.pure_opd_vision
        return False

    def _update_adaptive_coef(self, text_val_metrics: dict, vision_val_metrics: dict) -> None:
        """Update adaptive coef multipliers based on per-data-type val scores."""
        if not self.config.coev.opd_adaptive_coef:
            return
        # text as teacher: compare text branch vs vision branch on text data
        text_on_text = text_val_metrics.get("branch_text/val/text_reward_score", 0.0)
        vision_on_text = vision_val_metrics.get("branch_vision/val/text_reward_score", 0.0)
        self._adaptive_coef_text = 1.0 if text_on_text > vision_on_text else 0.0

        # vision as teacher: compare vision branch vs text branch on vision data
        vision_on_vision = vision_val_metrics.get("branch_vision/val/image_reward_score", 0.0)
        text_on_vision = text_val_metrics.get("branch_text/val/image_reward_score", 0.0)
        self._adaptive_coef_vision = 1.0 if vision_on_vision > text_on_vision else 0.0

        print(
            f"[adaptive_coef] text_on_text={text_on_text:.4f} vs vision_on_text={vision_on_text:.4f}"
            f" => text_teaches={'ON' if self._adaptive_coef_text else 'OFF'}"
            f" | vision_on_vision={vision_on_vision:.4f} vs text_on_vision={text_on_vision:.4f}"
            f" => vision_teaches={'ON' if self._adaptive_coef_vision else 'OFF'}",
            flush=True,
        )

    def _prepare_batch(
        self,
        branch: BranchState,
        batch: DataProto,
        metrics: dict[str, Any],
        scorer_branch: BranchState = None,
        metric_prefix: str = None,
        is_final_opd: bool = False,
    ) -> DataProto:
        """Compute reward, cross reward, log probs, and advantages — everything before the update."""
        prefix = metric_prefix if metric_prefix is not None else branch.name
        is_foreign = scorer_branch is not None
        use_task_reward = (not is_foreign) or self.config.coev.foreign_grpo_loss_enabled
        if "token_level_scores" not in batch.batch:
            if use_task_reward:
                reward_batch = batch.select(
                    batch_keys=["responses", "response_mask"],
                    non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
                )
                reward_tensor, reward_metrics = ray.get(branch.reward_fn.compute_reward.remote(reward_batch))
                batch.batch["token_level_scores"] = reward_tensor
                metrics.update(_prefix_metrics(prefix, {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}))
            else:
                batch.batch["token_level_scores"] = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
                metrics[f"{prefix}/coev/foreign_grpo_disabled"] = 1.0

        # Cross reward: driven by caller via scorer_branch parameter.
        # When OPD cycling is active, the caller already gates whether scorer_branch is passed,
        # so we only need the static flag when cycling is NOT configured.
        opd_cycling = self.config.coev.opd_off_steps > 0 or self.config.coev.opd_on_steps > 0
        cross_reward_enabled = is_final_opd or (True if opd_cycling else self.config.coev.cross_reward_loss_enabled)
        if scorer_branch is not None and cross_reward_enabled:
            self._compute_cross_reward(scorer_branch, batch, prefix, metrics)

        old_log_probs = branch.actor_rollout_ref_wg.compute_log_probs(batch)
        batch = batch.union(old_log_probs)

        if self.use_critic:
            values = branch.critic_wg.compute_values(batch)
            batch = batch.union(values)

        if not branch.config.algorithm.use_kl_loss and not branch.config.algorithm.disable_kl:
            batch, kl_metrics = apply_kl_penalty(batch, branch.kl_ctrl, branch.config.algorithm.kl_penalty)
            metrics.update(_prefix_metrics(prefix, kl_metrics))
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        batch = compute_advantage(
            batch,
            adv_estimator=branch.config.algorithm.adv_estimator,
            gamma=branch.config.algorithm.gamma,
            lam=branch.config.algorithm.lam,
        )

        # reverse_kl advantage mixing: token-level KL as advantage (similar to G-OPD)
        if "peer_log_probs" in batch.batch:
            reverse_kl_coef = self._get_reverse_kl_coef(branch)
            old_log_probs = batch.batch["old_log_probs"]
            peer_log_probs = batch.batch["peer_log_probs"]
            mask = batch.batch["response_mask"]
            reverse_kl = old_log_probs - peer_log_probs  # per-token KL(policy || peer)
            if is_final_opd:
                batch.batch["advantages"] = reverse_kl_coef * torch.clamp(-reverse_kl, min=0.0) * mask
            else:
                if self.config.coev.cross_reward_style == "reverse_kl_positive":
                    kl_advantage = torch.clamp(-reverse_kl, min=0.0) * mask
                else:
                    kl_advantage = -reverse_kl * mask  # token-level, no sequence averaging
                batch.batch["advantages"] = (
                    (1 - reverse_kl_coef) * batch.batch["advantages"] + reverse_kl_coef * kl_advantage
                )
            # logging: sequence-level mean for monitoring
            seq_kl = (reverse_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            metrics[f"{prefix}/coev/reverse_kl_mean"] = seq_kl.mean().item()
            metrics[f"{prefix}/coev/reverse_kl_std"] = seq_kl.std().item() if seq_kl.numel() > 1 else 0.0
            del batch.batch["peer_log_probs"]

        return batch

    def _foreign_update_has_signal(self, branch: BranchState) -> bool:
        """Return whether foreign samples can contribute actor gradients for *branch*."""
        if self.config.coev.foreign_grpo_loss_enabled:
            return True

        if self.config.coev.cross_reward_style == "reward_bonus":
            return self.config.coev.anchor_coef != 0.0

        return self._get_reverse_kl_coef(branch) != 0.0

    def _update_branch(
        self,
        branch: BranchState,
        batch: Optional[DataProto],
        metrics: dict[str, Any],
        step: int,
        metric_prefix: str = None,
    ) -> None:
        """Run critic and actor parameter updates on a prepared batch."""
        prefix = metric_prefix if metric_prefix is not None else branch.name
        if self._is_branch_frozen_for_cross_reward(branch):
            metrics[f"{prefix}/coev/frozen"] = 1.0
            metrics[f"{prefix}/coev/update_skipped"] = 1.0
            return
        if batch is None:
            metrics[f"{prefix}/coev/update_skipped"] = 1.0
            return
        if self.use_critic:
            critic_output = branch.critic_wg.update_critic(batch)
            metrics.update(_prefix_metrics(prefix, reduce_metrics(critic_output.non_tensor_batch)))

        if self.config.trainer.critic_warmup <= step:
            actor_output = branch.actor_rollout_ref_wg.update_actor(batch)
            metrics.update(_prefix_metrics(prefix, reduce_metrics(actor_output.non_tensor_batch)))


    def _validate_branch(self, branch: BranchState, dataloader: StatefulDataLoader, step: int = 0) -> dict[str, Any]:
        reward_tensor_lst = []
        data_type_lst = []
        problem_type_lst = []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []

        print(f"Start validation for {branch.name}...", flush=True)
        t0 = time.time()
        print(f"  [{branch.name}] calling prepare_rollout_engine()...", flush=True)
        branch.actor_rollout_ref_wg.prepare_rollout_engine()
        print(f"  [{branch.name}] prepare_rollout_engine() done in {time.time() - t0:.1f}s", flush=True)
        val_batch_idx = 0
        for batch_dict in dataloader:
            t_batch = time.time()
            print(f"  [{branch.name}] val batch {val_batch_idx}: preparing...", flush=True)
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = branch.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = branch.config.worker.rollout.val_override_config
            test_gen_batch.meta_info.update(self._build_meta_info())

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, branch.actor_rollout_ref_wg.world_size)
            print(f"  [{branch.name}] val batch {val_batch_idx}: generating (bsz={len(test_gen_batch)})...", flush=True)
            test_output_gen_batch = branch.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            print(f"  [{branch.name}] val batch {val_batch_idx}: generated in {time.time() - t_batch:.1f}s", flush=True)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)
            val_batch_idx += 1

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_batch = test_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in test_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            print(f"  [{branch.name}] val batch {val_batch_idx}: rewarding", flush=True)
            reward_tensor, reward_metrics = ray.get(branch.val_reward_fn.compute_reward.remote(reward_batch))
            print(f"  [{branch.name}] val batch {val_batch_idx}: reward in {time.time() - t_batch:.1f}s", flush=True)

            reward_tensor_lst.append(reward_tensor)
            if "data_type" in test_batch.non_tensor_batch:
                data_type_lst.extend(test_batch.non_tensor_batch["data_type"].tolist())
            if "problem_type" in test_batch.non_tensor_batch:
                problem_type_lst.extend(test_batch.non_tensor_batch["problem_type"].tolist())
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)
            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

            if self.config.trainer.val_generations_to_log > 0:
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch["prompts"]]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in test_batch.batch["responses"]]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_inputs.extend(input_texts)
                sample_outputs.extend(output_texts)
                sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
                sample_scores.extend(scores)

        branch.actor_rollout_ref_wg.release_rollout_engine()
        if self.config.trainer.val_generations_to_log > 0 and sample_inputs:
            samples = list(zip(sample_inputs, sample_outputs, sample_labels, sample_scores, [branch.name] * len(sample_inputs)))
            with self._logger_lock:
                self.logger.log_generation(samples[: self.config.trainer.val_generations_to_log], step)

        all_reward_tensor = torch.cat(reward_tensor_lst, dim=0)
        reward_score = all_reward_tensor.sum(-1).mean().item()
        metrics = {
            "val/reward_score": reward_score,
            **{f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()},
            **{f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()},
        }

        def _add_group_metrics(group_type_lst: list[str]) -> None:
            per_type_scores = defaultdict(list)
            per_type_reward_metrics = defaultdict(lambda: defaultdict(list))
            sample_scores_all = all_reward_tensor.sum(-1).tolist()
            for i, dt in enumerate(group_type_lst):
                per_type_scores[dt].append(sample_scores_all[i])
                for key, values in reward_metrics_lst.items():
                    per_type_reward_metrics[dt][key].append(values[i])
            for dt, scores in per_type_scores.items():
                metrics[f"val/{dt}_reward_score"] = sum(scores) / len(scores)
            for dt, dt_metrics in per_type_reward_metrics.items():
                for key, values in dt_metrics.items():
                    metrics[f"val/{dt}_{key}_reward"] = sum(values) / len(values)

        valid_data_type = data_type_lst and len(data_type_lst) == all_reward_tensor.shape[0]
        valid_problem_type = problem_type_lst and len(problem_type_lst) == all_reward_tensor.shape[0]
        val_metric_grouping = self.config.trainer.val_metric_grouping

        if val_metric_grouping == "auto":
            # Prefer problem_type grouping when the entire validation set has the same data_type.
            group_type_lst = data_type_lst
            if (
                valid_data_type
                and len(set(data_type_lst)) == 1
                and valid_problem_type
            ):
                group_type_lst = problem_type_lst
            if group_type_lst and len(group_type_lst) == all_reward_tensor.shape[0]:
                _add_group_metrics(group_type_lst)
        else:
            if val_metric_grouping in ("data_type", "both") and valid_data_type:
                _add_group_metrics(data_type_lst)
            if val_metric_grouping in ("problem_type", "both") and valid_problem_type:
                _add_group_metrics(problem_type_lst)

        print(f"Finish validation for {branch.name}: {convert_dict_to_str(metrics)}")
        return _prefix_metrics(branch.name, metrics)




    def _setup_wandb_metrics(self):
        """W&B metric axes are configured by the shared logger."""
        try:
            pass
        except Exception:
            pass

    def fit(self):
        t_fit = time.time()
        print(f"[fit] Creating logger...", flush=True)
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        print(f"[fit] Logger created in {time.time() - t_fit:.1f}s", flush=True)
        self._setup_wandb_metrics()

        t_ckpt = time.time()
        print(f"[fit] Loading checkpoint...", flush=True)
        self._load_checkpoint()
        print(f"[fit] Checkpoint loaded in {time.time() - t_ckpt:.1f}s", flush=True)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = {}
            t_val = time.time()
            print(f"[fit] val_before_train: starting parallel validation...", flush=True)
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_vt = executor.submit(
                    self._validate_branch, self.branch_text, self.text_val_dataloader, step=self.text_step
                )
                future_vv = executor.submit(
                    self._validate_branch, self.branch_vision, self.vision_val_dataloader, step=self.vision_step
                )
                val_metrics.update(future_vt.result())
                val_metrics.update(future_vv.result())
            print(f"[fit] val_before_train done in {time.time() - t_val:.1f}s", flush=True)
            self.text_val_reward_score = val_metrics.get("branch_text/val/reward_score", 0.0)
            self.vision_val_reward_score = val_metrics.get("branch_vision/val/reward_score", 0.0)
            self._update_adaptive_coef(val_metrics, val_metrics)
            # Log both branches in a single call so wandb global step stays in sync
            combined_val = dict(val_metrics)
            combined_val["branch_text_step"] = self.text_step
            combined_val["branch_vision_step"] = self.vision_step
            self.logger.log(data=combined_val)
            if self.config.trainer.val_only:
                return

        self._fit_cross_reward()

        # Final checkpoint with both branches
        self._save_checkpoint()


    def _fit_cross_reward(self):
        """Synchronized cross-reward training.

        Each step, both branches see the **same** text batch and the **same** vision batch:
        - text branch: text batch → normal GRPO, vision batch → GRPO + KL(text || vision)
        - vision branch: vision batch → normal GRPO, text batch → GRPO + KL(vision || text)

        Rollouts and updates run in parallel across branches. Teacher scoring and
        advantage preparation complete before the update phase starts.
        """
        val_freq = self.config.trainer.val_freq
        save_freq = self.config.trainer.save_freq
        text_iter_attr = "text_data_iterator"
        vision_iter_attr = "vision_data_iterator"
        setattr(self, text_iter_attr, iter(self.text_train_dataloader))
        setattr(self, vision_iter_attr, iter(self.vision_train_dataloader))
        if self.config.coev.freeze_zero_opd_branches:
            frozen_names = [
                branch.name
                for branch in (self.branch_text, self.branch_vision)
                if self._is_branch_frozen_for_cross_reward(branch)
            ]
            print(
                "[cross_reward] freeze_zero_opd_branches enabled; initial frozen branches: "
                + (", ".join(frozen_names) if frozen_names else "none"),
                flush=True,
            )
        pure_opd_names = [
            branch.name
            for branch in (self.branch_text, self.branch_vision)
            if self._is_branch_pure_opd_for_cross_reward(branch)
        ]
        if pure_opd_names:
            print(
                "[cross_reward] pure OPD branches skip native data: " + ", ".join(pure_opd_names),
                flush=True,
            )

        while self.text_step < self.training_steps:
            step = self.text_step + 1
            self.text_step = step
            self.vision_step = step  # synchronized steps
            text_metrics: dict[str, Any] = {}
            vision_metrics: dict[str, Any] = {}
            text_timing: dict[str, float] = {}
            vision_timing: dict[str, float] = {}
            text_frozen = self._is_branch_frozen_for_cross_reward(self.branch_text)
            vision_frozen = self._is_branch_frozen_for_cross_reward(self.branch_vision)
            if text_frozen and vision_frozen:
                raise RuntimeError("Both cross-reward branches are frozen; no branch can be optimized.")
            text_pure_opd = self._is_branch_pure_opd_for_cross_reward(self.branch_text)
            vision_pure_opd = self._is_branch_pure_opd_for_cross_reward(self.branch_vision)
            text_metrics["branch_text/coev/frozen"] = float(text_frozen)
            vision_metrics["branch_vision/coev/frozen"] = float(vision_frozen)
            text_metrics["branch_text/coev/pure_opd"] = float(text_pure_opd)
            vision_metrics["branch_vision/coev/pure_opd"] = float(vision_pure_opd)
            text_native = text_foreign = text_combined = None
            vision_native = vision_foreign = vision_combined = None

            final_opd_active = self._is_final_opd(step)
            opd_active = self._is_opd_active(step) or final_opd_active

            text_batch_dict = None
            vision_batch_dict = None
            if opd_active:
                need_text_batch = (not text_frozen and not text_pure_opd) or (not vision_frozen)
                need_vision_batch = (not vision_frozen and not vision_pure_opd) or (not text_frozen)
            else:
                need_text_batch = not text_frozen and not text_pure_opd
                need_vision_batch = not vision_frozen and not vision_pure_opd
            if need_text_batch:
                text_batch_dict = self._next_batch_dict(text_iter_attr, self.text_train_dataloader)
            if need_vision_batch:
                vision_batch_dict = self._next_batch_dict(vision_iter_attr, self.vision_train_dataloader)

            if opd_active:
                # ========== OPD ON: merged rollout (native+foreign), one generate call ==========
                def _rollout_branch_merged(branch, native_dict, foreign_dict, metrics, timing, include_native: bool):
                    """Merge native+foreign into one batch, rollout once, then split."""
                    with timer("gen", timing):
                        foreign_proto = DataProto.from_single_dict(foreign_dict, meta_info=self._build_meta_info())
                        foreign_size = len(foreign_proto)
                        if include_native:
                            native_proto = DataProto.from_single_dict(native_dict, meta_info=self._build_meta_info())
                            native_size = len(native_proto)
                            self._align_non_tensor_keys(native_proto, foreign_proto)
                            merged = DataProto.concat([native_proto, foreign_proto])
                            is_native = np.array([1] * native_size + [0] * foreign_size, dtype=np.int32)
                        else:
                            native_size = 0
                            merged = foreign_proto
                            is_native = np.array([0] * foreign_size, dtype=np.int32)
                        merged.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(merged.batch))], dtype=object,
                        )
                        merged.non_tensor_batch["_is_native"] = is_native

                        gen_batch = self._make_generation_batch(merged)
                        branch.actor_rollout_ref_wg.prepare_rollout_engine()
                        gen_output = branch.actor_rollout_ref_wg.generate_sequences(gen_batch)
                        branch.actor_rollout_ref_wg.release_rollout_engine()

                        n = self.config.worker.rollout.n
                        merged = merged.repeat(repeat_times=n, interleave=True)
                        merged = merged.union(gen_output)

                        if self.config.algorithm.online_filtering:
                            all_metrics = defaultdict(list)
                            merged = self._apply_online_filtering(merged, all_metrics, branch.reward_fn)
                            metrics.update(
                                _prefix_metrics(branch.name, {f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                            )

                        rollout_batch_size = branch.config.data.rollout_batch_size
                        native_mask = merged.non_tensor_batch["_is_native"] == 1
                        native_idxs = np.where(native_mask)[0][: rollout_batch_size * n]
                        foreign_idxs = np.where(~native_mask)[0][: rollout_batch_size * n]
                        native_batch = merged[native_idxs] if native_size > 0 else None
                        foreign_batch = merged[foreign_idxs]
                        if native_batch is not None:
                            del native_batch.non_tensor_batch["_is_native"]
                        del foreign_batch.non_tensor_batch["_is_native"]
                    return native_batch, foreign_batch

                rollout_jobs = []
                if not text_frozen:
                    rollout_jobs.append((
                        "text",
                        self.branch_text,
                        text_batch_dict,
                        vision_batch_dict,
                        text_metrics,
                        text_timing,
                        not text_pure_opd,
                    ))
                    if text_pure_opd:
                        text_metrics["branch_text/coev/native_rollout_skipped"] = 1.0
                else:
                    text_metrics["branch_text/coev/rollout_skipped"] = 1.0
                if not vision_frozen:
                    rollout_jobs.append((
                        "vision",
                        self.branch_vision,
                        vision_batch_dict,
                        text_batch_dict,
                        vision_metrics,
                        vision_timing,
                        not vision_pure_opd,
                    ))
                    if vision_pure_opd:
                        vision_metrics["branch_vision/coev/native_rollout_skipped"] = 1.0
                else:
                    vision_metrics["branch_vision/coev/rollout_skipped"] = 1.0

                with ThreadPoolExecutor(max_workers=len(rollout_jobs)) as executor:
                    future_map = {
                        executor.submit(
                            _rollout_branch_merged,
                            branch,
                            native_dict,
                            foreign_dict,
                            metrics,
                            timing,
                            include_native,
                        ): branch_key
                        for branch_key, branch, native_dict, foreign_dict, metrics, timing, include_native in rollout_jobs
                    }
                    for future, branch_key in future_map.items():
                        native_batch, foreign_batch = future.result()
                        if branch_key == "text":
                            text_native, text_foreign = native_batch, foreign_batch
                        else:
                            vision_native, vision_foreign = native_batch, foreign_batch

                # Balance all four batches
                for batch, wg, metrics, prefix in [
                    (text_native, self.branch_text.actor_rollout_ref_wg, text_metrics, "branch_text"),
                    (text_foreign, self.branch_text.actor_rollout_ref_wg, text_metrics, "branch_text/foreign"),
                    (vision_native, self.branch_vision.actor_rollout_ref_wg, vision_metrics, "branch_vision"),
                    (vision_foreign, self.branch_vision.actor_rollout_ref_wg, vision_metrics, "branch_vision/foreign"),
                ]:
                    if batch is None:
                        continue
                    self._balance_batch(batch, wg.world_size, metrics, logging_prefix=prefix)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # Default coev metrics
                text_metrics["branch_text/foreign/coev/cross_reward_applied"] = 0.0
                text_metrics["branch_text/foreign/coev/cross_reward_mean"] = 0.0
                text_metrics["branch_text/foreign/coev/cross_reward_std"] = 0.0
                vision_metrics["branch_vision/foreign/coev/cross_reward_applied"] = 0.0
                vision_metrics["branch_vision/foreign/coev/cross_reward_mean"] = 0.0
                vision_metrics["branch_vision/foreign/coev/cross_reward_std"] = 0.0

                # Prepare Phase 1: native data (parallel)
                def _prep_text_native():
                    with timer("prepare_native", text_timing):
                        return self._prepare_batch(
                            self.branch_text, text_native, text_metrics, is_final_opd=final_opd_active
                        )

                def _prep_vision_native():
                    with timer("prepare_native", vision_timing):
                        return self._prepare_batch(
                            self.branch_vision, vision_native, vision_metrics, is_final_opd=final_opd_active
                        )

                native_prepare_jobs = []
                if not text_frozen and not text_pure_opd and text_native is not None:
                    native_prepare_jobs.append(("text", _prep_text_native))
                if not vision_frozen and not vision_pure_opd and vision_native is not None:
                    native_prepare_jobs.append(("vision", _prep_vision_native))
                if native_prepare_jobs:
                    with ThreadPoolExecutor(max_workers=len(native_prepare_jobs)) as executor:
                        future_map = {
                            executor.submit(fn): branch_key
                            for branch_key, fn in native_prepare_jobs
                        }
                        for future, branch_key in future_map.items():
                            if branch_key == "text":
                                text_native = future.result()
                            else:
                                vision_native = future.result()

                # Prepare Phase 2: foreign data with cross-reward (parallel)
                def _prep_text_foreign():
                    with timer("prepare_foreign", text_timing):
                        return self._prepare_batch(
                            self.branch_text, text_foreign, text_metrics,
                            scorer_branch=self.branch_vision,
                            metric_prefix="branch_text/foreign",
                            is_final_opd=final_opd_active,
                        )

                def _prep_vision_foreign():
                    with timer("prepare_foreign", vision_timing):
                        return self._prepare_batch(
                            self.branch_vision, vision_foreign, vision_metrics,
                            scorer_branch=self.branch_text,
                            metric_prefix="branch_vision/foreign",
                            is_final_opd=final_opd_active,
                        )

                foreign_prepare_jobs = []
                if not text_frozen and text_foreign is not None:
                    foreign_prepare_jobs.append(("text", _prep_text_foreign))
                if not vision_frozen and vision_foreign is not None:
                    foreign_prepare_jobs.append(("vision", _prep_vision_foreign))
                if foreign_prepare_jobs:
                    with ThreadPoolExecutor(max_workers=len(foreign_prepare_jobs)) as executor:
                        future_map = {
                            executor.submit(fn): branch_key
                            for branch_key, fn in foreign_prepare_jobs
                        }
                        for future, branch_key in future_map.items():
                            if branch_key == "text":
                                text_foreign = future.result()
                            else:
                                vision_foreign = future.result()

                text_foreign_enabled = (not text_frozen) and self._foreign_update_has_signal(self.branch_text)
                vision_foreign_enabled = (not vision_frozen) and self._foreign_update_has_signal(self.branch_vision)
                text_metrics["branch_text/foreign/coev/update_skipped"] = 0.0 if text_foreign_enabled else 1.0
                vision_metrics["branch_vision/foreign/coev/update_skipped"] = 0.0 if vision_foreign_enabled else 1.0

                if final_opd_active:
                    text_combined = text_foreign if text_foreign_enabled else None
                    vision_combined = vision_foreign if vision_foreign_enabled else None
                else:
                    if text_frozen:
                        text_combined = None
                    elif text_pure_opd:
                        text_combined = text_foreign if text_foreign_enabled else None
                    elif text_foreign_enabled:
                        self._align_non_tensor_keys(text_native, text_foreign)
                        text_combined = DataProto.concat([text_native, text_foreign])
                        text_combined.meta_info["global_token_num"] = (
                            text_native.meta_info.get("global_token_num", [])
                            + text_foreign.meta_info.get("global_token_num", [])
                        )
                    else:
                        text_combined = text_native

                    if vision_frozen:
                        vision_combined = None
                    elif vision_pure_opd:
                        vision_combined = vision_foreign if vision_foreign_enabled else None
                    elif vision_foreign_enabled:
                        self._align_non_tensor_keys(vision_native, vision_foreign)
                        vision_combined = DataProto.concat([vision_native, vision_foreign])
                        vision_combined.meta_info["global_token_num"] = (
                            vision_native.meta_info.get("global_token_num", [])
                            + vision_foreign.meta_info.get("global_token_num", [])
                        )
                    else:
                        vision_combined = vision_native
            else:
                # ========== OPD OFF: native-only rollout, no foreign data ==========
                def _rollout_native_only(branch, native_dict, metrics, timing):
                    with timer("gen", timing):
                        branch.actor_rollout_ref_wg.prepare_rollout_engine()
                        batch = self._rollout_from_batch_dict(branch, native_dict, metrics)
                        branch.actor_rollout_ref_wg.release_rollout_engine()
                        return batch

                rollout_jobs = []
                if not text_frozen and not text_pure_opd:
                    rollout_jobs.append((
                        "text",
                        self.branch_text,
                        text_batch_dict,
                        text_metrics,
                        text_timing,
                    ))
                else:
                    text_metrics["branch_text/coev/rollout_skipped"] = 1.0
                    if text_pure_opd:
                        text_metrics["branch_text/coev/native_rollout_skipped"] = 1.0
                if not vision_frozen and not vision_pure_opd:
                    rollout_jobs.append((
                        "vision",
                        self.branch_vision,
                        vision_batch_dict,
                        vision_metrics,
                        vision_timing,
                    ))
                else:
                    vision_metrics["branch_vision/coev/rollout_skipped"] = 1.0
                    if vision_pure_opd:
                        vision_metrics["branch_vision/coev/native_rollout_skipped"] = 1.0

                if rollout_jobs:
                    with ThreadPoolExecutor(max_workers=len(rollout_jobs)) as executor:
                        future_map = {
                            executor.submit(
                                _rollout_native_only,
                                branch,
                                native_dict,
                                metrics,
                                timing,
                            ): branch_key
                            for branch_key, branch, native_dict, metrics, timing in rollout_jobs
                        }
                        for future, branch_key in future_map.items():
                            if branch_key == "text":
                                text_native = future.result()
                            else:
                                vision_native = future.result()

                # Balance native batches only
                for batch, wg, metrics, prefix in [
                    (text_native, self.branch_text.actor_rollout_ref_wg, text_metrics, "branch_text"),
                    (vision_native, self.branch_vision.actor_rollout_ref_wg, vision_metrics, "branch_vision"),
                ]:
                    if batch is None:
                        continue
                    self._balance_batch(batch, wg.world_size, metrics, logging_prefix=prefix)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # Prepare native data only (parallel)
                def _prep_text_native():
                    with timer("prepare_native", text_timing):
                        return self._prepare_batch(
                            self.branch_text, text_native, text_metrics, is_final_opd=final_opd_active
                        )

                def _prep_vision_native():
                    with timer("prepare_native", vision_timing):
                        return self._prepare_batch(
                            self.branch_vision, vision_native, vision_metrics, is_final_opd=final_opd_active
                        )

                native_prepare_jobs = []
                if not text_frozen and not text_pure_opd and text_native is not None:
                    native_prepare_jobs.append(("text", _prep_text_native))
                if not vision_frozen and not vision_pure_opd and vision_native is not None:
                    native_prepare_jobs.append(("vision", _prep_vision_native))
                if native_prepare_jobs:
                    with ThreadPoolExecutor(max_workers=len(native_prepare_jobs)) as executor:
                        future_map = {
                            executor.submit(fn): branch_key
                            for branch_key, fn in native_prepare_jobs
                        }
                        for future, branch_key in future_map.items():
                            if branch_key == "text":
                                text_native = future.result()
                            else:
                                vision_native = future.result()

                text_combined = None if (text_frozen or text_pure_opd) else text_native
                vision_combined = None if (vision_frozen or vision_pure_opd) else vision_native

            # ========== Update (shared for both paths) ==========
            def _update_text():
                with timer("update", text_timing):
                    self._update_branch(self.branch_text, text_combined, text_metrics, step)

            def _update_vision():
                with timer("update", vision_timing):
                    self._update_branch(self.branch_vision, vision_combined, vision_metrics, step)

            with ThreadPoolExecutor(max_workers=2) as executor:
                ft = executor.submit(_update_text)
                fv = executor.submit(_update_vision)
                ft.result()
                fv.result()

            # ========== Sync ==========

            # ========== Validate (parallel) ==========
            if self.val_reward_fn is not None and val_freq > 0 and step % val_freq == 0:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    ft = executor.submit(
                        self._validate_branch, self.branch_text,
                        self.text_val_dataloader, step=step,
                    )
                    fv = executor.submit(
                        self._validate_branch, self.branch_vision,
                        self.vision_val_dataloader, step=step,
                    )
                    text_metrics.update(ft.result())
                    vision_metrics.update(fv.result())
                self.text_val_reward_score = text_metrics.get("branch_text/val/reward_score", 0.0)
                self.vision_val_reward_score = vision_metrics.get("branch_vision/val/reward_score", 0.0)
                self._update_adaptive_coef(text_metrics, vision_metrics)

            # ========== Checkpoint (parallel) ==========
            if save_freq > 0 and step % save_freq == 0:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    ft = executor.submit(
                        self._save_branch_checkpoint,
                        self.branch_text, step, self.text_train_dataloader, "text",
                    )
                    fv = executor.submit(
                        self._save_branch_checkpoint,
                        self.branch_vision, step, self.vision_train_dataloader, "vision",
                    )
                    ft.result()
                    fv.result()

            # ========== Log (single call so wandb global step = actual step) ==========
            combined_metrics: dict[str, Any] = {}
            text_log_batch = text_native if text_native is not None else text_foreign
            vision_log_batch = vision_native if vision_native is not None else vision_foreign
            for branch, batch, metrics, timing in [
                (self.branch_text, text_log_batch, text_metrics, text_timing),
                (self.branch_vision, vision_log_batch, vision_metrics, vision_timing),
            ]:
                if batch is not None:
                    metrics.update(_prefix_metrics(branch.name, compute_data_metrics(batch=batch, use_critic=self.use_critic)))
                else:
                    metrics[f"{branch.name}/coev/rollout_skipped"] = 1.0
                metrics.update({f"{branch.name}/timing_s/{key}": value for key, value in timing.items()})
                metrics[f"{branch.name}/coev/reverse_kl_coef"] = self._get_reverse_kl_coef(branch)
                metrics.setdefault(f"{branch.name}/coev/update_skipped", 0.0)
                metrics[f"{branch.name}_step"] = step
                combined_metrics.update(metrics)
            combined_metrics["coev/anchor_coef"] = self.config.coev.anchor_coef
            combined_metrics["coev/foreign_grpo_loss_enabled"] = float(self.config.coev.foreign_grpo_loss_enabled)
            combined_metrics["coev/cross_reward_loss_enabled"] = float(self.config.coev.cross_reward_loss_enabled)
            combined_metrics["coev/opd_active"] = float(opd_active)
            combined_metrics["coev/force_final_opd"] = float(self.config.coev.force_final_opd)
            combined_metrics["coev/freeze_zero_opd_branches"] = float(self.config.coev.freeze_zero_opd_branches)
            combined_metrics["coev/pure_opd_text"] = float(self.config.coev.pure_opd_text)
            combined_metrics["coev/pure_opd_vision"] = float(self.config.coev.pure_opd_vision)
            combined_metrics["coev/adaptive_coef_text"] = self._adaptive_coef_text
            combined_metrics["coev/adaptive_coef_vision"] = self._adaptive_coef_vision
            self.logger.log(data=combined_metrics)

            print(f"[cross_reward] step {step}/{self.training_steps} done. (opd={'ON' if opd_active else 'OFF'})")

    def _rollout_from_batch_dict(
        self, branch: BranchState, batch_dict: dict, metrics: dict[str, Any],
    ) -> DataProto:
        """Rollout a branch on a pre-fetched batch_dict (no dataloader iteration)."""
        all_metrics = defaultdict(list)
        new_batch = DataProto.from_single_dict(batch_dict, meta_info=self._build_meta_info())
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object,
        )

        gen_batch = self._make_generation_batch(new_batch)
        gen_output = branch.actor_rollout_ref_wg.generate_sequences(gen_batch)

        new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
        new_batch = new_batch.union(gen_output)

        if self.config.algorithm.online_filtering:
            new_batch = self._apply_online_filtering(new_batch, all_metrics, branch.reward_fn)
            metrics.update(
                _prefix_metrics(branch.name, {f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
            )

        rollout_batch_size = branch.config.data.rollout_batch_size
        return new_batch[: rollout_batch_size * self.config.worker.rollout.n]

