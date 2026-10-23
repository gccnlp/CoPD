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
Three-branch CoPD trainer with native GRPO and on-policy distillation.

With ``coev.hub_branch="text"``, the text branch is the hub and the image
and video branches are spokes. Distillation is limited to hub-spoke pairs.
Each branch maintains its own actor/rollout workers, validation, and checkpoints.
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

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
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
class TriBranchSpec:
    name: str
    train_dataloader: StatefulDataLoader
    val_dataloader: StatefulDataLoader
    config: Optional[PPOConfig] = None
    reward_fn: Optional[AutoRewardManager] = None
    val_reward_fn: Optional[AutoRewardManager] = None
    rollout_seed: Optional[int] = None
    rollout_temperature: Optional[float] = None
    rollout_top_p: Optional[float] = None
    rollout_batch_size: Optional[int] = None
    actor_global_batch_size: Optional[int] = None
    critic_global_batch_size: Optional[int] = None
    model_path: Optional[str] = None
    reverse_kl_coef: Optional[float] = None


@dataclass
class TriBranchState:
    name: str
    config: PPOConfig
    train_dataloader: StatefulDataLoader
    val_dataloader: StatefulDataLoader
    actor_rollout_ref_wg: RayWorkerGroup
    critic_wg: Optional[RayWorkerGroup]
    kl_ctrl: Any
    reward_fn: Any = None
    val_reward_fn: Any = None
    reverse_kl_coef: Optional[float] = None


class TriBranchCoEvolRayPPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        branch_specs: list[TriBranchSpec],
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.branch_specs = list(branch_specs)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        if len(self.branch_specs) != 3:
            raise ValueError("TriBranchCoEvolRayPPOTrainer requires exactly three branch specs.")
        branch_names = [spec.name for spec in self.branch_specs]
        if len(set(branch_names)) != len(branch_names):
            raise ValueError(f"Branch names must be unique, got: {branch_names}")
        self.hub_branch_name = self._resolve_hub_branch_name()

        if not self.config.coev.enabled:
            raise ValueError("TriBranchCoEvolRayPPOTrainer requires `coev.enabled=true`.")
        if not self.config.parallel.enabled:
            raise ValueError("TriBranchCoEvolRayPPOTrainer requires `parallel.enabled=true`.")
        if self.config.coev.mode != "cross_reward":
            raise NotImplementedError(
                "TriBranchCoEvolRayPPOTrainer currently supports only `coev.mode=cross_reward`."
            )
        if self.config.coev.cross_reward_style not in ("reward_bonus", "reverse_kl", "reverse_kl_positive"):
            raise ValueError(
                f"Unsupported cross_reward_style: {self.config.coev.cross_reward_style}. "
                "Use 'reward_bonus', 'reverse_kl', or 'reverse_kl_positive'."
            )
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("TriBranchCoEvolRayPPOTrainer does not support REMAX yet.")
        if self.config.coev.opd_adaptive_coef:
            print(
                "[tri-branch] `coev.opd_adaptive_coef=true` is not implemented for three branches; "
                "falling back to static reverse_kl coefficients.",
                flush=True,
            )

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
        else:
            self.training_steps = max(len(spec.train_dataloader) for spec in self.branch_specs) * self.config.trainer.total_epochs

        self.branch_count = len(self.branch_specs)
        self.node_allocation = self._resolve_node_allocation()
        self.branch_roles, self.branch_gpu_counts = self._resolve_branch_gpu_counts()

        self.branch_prepared: list[tuple[TriBranchSpec, PPOConfig]] = []
        for idx, spec in enumerate(self.branch_specs):
            branch_config = deepcopy(spec.config or self.config)
            self._apply_branch_overrides(branch_config, spec, idx)
            self._prepare_branch_optim_steps(branch_config)
            self.branch_prepared.append((spec, branch_config))
        self._validate_branch_distribution()

        self.branches: list[TriBranchState] = []
        self.branch_map: dict[str, TriBranchState] = {}
        self.branch_steps = {spec.name: 0 for spec in self.branch_specs}
        self.branch_val_reward_scores = {spec.name: 0.0 for spec in self.branch_specs}
        self.best_branch_val_reward_scores = {spec.name: -1.0 for spec in self.branch_specs}
        self.best_branch_steps = {spec.name: None for spec in self.branch_specs}

        self.global_step = 0
        self._tracker_lock = threading.Lock()
        self._logger_lock = threading.Lock()
        self._resume_warned = False
        self.wg_dicts = []

    def _infer_branch_role(self, spec: TriBranchSpec, index: int) -> str:
        normalized_name = spec.name.lower()
        if "text" in normalized_name:
            return "text"
        if "vision" in normalized_name or "image" in normalized_name:
            return "vision"
        if "video" in normalized_name:
            return "video"
        if index == 0:
            return "text"
        if index == 1:
            return "vision"
        return "video"

    def _resolve_hub_branch_name(self) -> Optional[str]:
        hub_branch = self.config.coev.hub_branch
        if hub_branch is None:
            return None
        hub_branch = hub_branch.strip()
        if not hub_branch:
            return None

        normalized_hub = hub_branch.lower()
        exact_matches = [spec.name for spec in self.branch_specs if spec.name == hub_branch]
        if len(exact_matches) == 1:
            return exact_matches[0]

        normalized_name_matches = [spec.name for spec in self.branch_specs if spec.name.lower() == normalized_hub]
        if len(normalized_name_matches) == 1:
            return normalized_name_matches[0]

        role_matches = [
            spec.name
            for index, spec in enumerate(self.branch_specs)
            if self._infer_branch_role(spec, index) == normalized_hub
        ]
        if len(role_matches) == 1:
            return role_matches[0]

        available = ", ".join(spec.name for spec in self.branch_specs)
        raise ValueError(
            "`coev.hub_branch` must match exactly one tri-branch by branch name or role "
            f"(`text`, `vision`, `video`). Got `{hub_branch}` with available branches: {available}."
        )

    def _get_connected_foreign_branch_names(self, branch_name: str) -> list[str]:
        all_branch_names = [spec.name for spec in self.branch_specs]
        if self.hub_branch_name is None:
            return [name for name in all_branch_names if name != branch_name]
        if branch_name == self.hub_branch_name:
            return [name for name in all_branch_names if name != branch_name]
        return [self.hub_branch_name]

    def _get_rollout_source_names(self, branch_name: str) -> list[str]:
        source_names = [branch_name, *self._get_connected_foreign_branch_names(branch_name)]
        return list(dict.fromkeys(source_names))

    def _resolve_node_allocation(self) -> Optional[dict[str, list[int]]]:
        """Build per-branch ``process_on_nodes`` when machine-level allocation is configured.

        Returns None when no ``parallel.nodes_*`` fields are set (colocated mode).
        """
        cfg = self.config.parallel
        nodes_map = {"text": cfg.nodes_text, "vision": cfg.nodes_vision, "video": cfg.nodes_video}
        if all(v is None for v in nodes_map.values()):
            return None

        nnodes = self.config.trainer.nnodes
        gpus_per_node = self.config.trainer.n_gpus_per_node

        role_to_spec: dict[str, str] = {}
        for index, spec in enumerate(self.branch_specs):
            role = self._infer_branch_role(spec, index)
            role_to_spec[role] = spec.name

        assigned: dict[str, int] = {}
        for role in ("text", "vision", "video"):
            n = nodes_map.get(role)
            if n is None:
                raise ValueError(
                    f"Machine-level allocation requires all of parallel.nodes_text, "
                    f"parallel.nodes_vision, parallel.nodes_video to be set. "
                    f"parallel.nodes_{role} is None."
                )
            if n <= 0:
                raise ValueError(f"parallel.nodes_{role} must be positive, got {n}.")
            assigned[role] = n

        total_nodes = sum(assigned.values())
        if total_nodes > nnodes:
            raise ValueError(
                f"Machine-level allocation needs {total_nodes} nodes "
                f"(text={assigned['text']}, vision={assigned['vision']}, video={assigned['video']}), "
                f"but trainer.nnodes={nnodes}."
            )
        if total_nodes < nnodes:
            print(
                f"[tri-branch] Warning: {nnodes - total_nodes} node(s) will be idle. "
                f"Allocated {total_nodes} out of {nnodes}.",
                flush=True,
            )

        allocation: dict[str, list[int]] = {}
        offset = 0
        for role in ("text", "vision", "video"):
            spec_name = role_to_spec[role]
            n = assigned[role]
            proc = [gpus_per_node] * n
            allocation[spec_name] = proc
            offset += n

        print(
            f"[tri-branch] Machine-level node allocation: "
            + ", ".join(f"{name}={sum(1 for x in proc if x > 0)} nodes" for name, proc in allocation.items()),
            flush=True,
        )
        return allocation

    def _resolve_branch_gpu_counts(self) -> tuple[dict[str, str], dict[str, int]]:
        branch_roles: dict[str, str] = {}
        branch_gpu_counts: dict[str, int] = {}

        if self.node_allocation is not None:
            gpus_per_node = self.config.trainer.n_gpus_per_node
            for index, spec in enumerate(self.branch_specs):
                role = self._infer_branch_role(spec, index)
                branch_roles[spec.name] = role
                branch_gpu_counts[spec.name] = gpus_per_node
            return branch_roles, branch_gpu_counts

        fallback = self.config.parallel.n_gpus_per_model
        if fallback is None:
            fallback = self.config.trainer.n_gpus_per_node // self.branch_count
        hub_gpu_count = self.config.parallel.n_gpus_per_model_hub
        leaf_gpu_count = self.config.parallel.n_gpus_per_model_leaf
        if self.hub_branch_name is not None:
            if hub_gpu_count is None:
                hub_gpu_count = 4
            if leaf_gpu_count is None:
                leaf_gpu_count = 2

        role_specific_counts = {
            "text": self.config.parallel.n_gpus_per_model_text,
            "vision": self.config.parallel.n_gpus_per_model_vision,
            "video": self.config.parallel.n_gpus_per_model_video,
        }

        for index, spec in enumerate(self.branch_specs):
            role = self._infer_branch_role(spec, index)
            if self.hub_branch_name is not None:
                gpu_count = hub_gpu_count if spec.name == self.hub_branch_name else leaf_gpu_count
            else:
                gpu_count = role_specific_counts.get(role)
                if gpu_count is None:
                    gpu_count = fallback
            if gpu_count is None or gpu_count <= 0:
                raise ValueError(
                    f"GPU allocation for tri-branch `{spec.name}` resolved to {gpu_count}. "
                    "Please set `parallel.n_gpus_per_model_hub` / `parallel.n_gpus_per_model_leaf`, "
                    "`parallel.n_gpus_per_model_<branch>`, or `parallel.n_gpus_per_model`."
                )
            branch_roles[spec.name] = role
            branch_gpu_counts[spec.name] = gpu_count

        return branch_roles, branch_gpu_counts

    @staticmethod
    def _validate_role_batch_geometry(
        role_name: str,
        role_config: Any,
        rollout_n: int,
        world_size: int,
        branch_name: str,
    ) -> None:
        ulysses_size = role_config.ulysses_size
        if ulysses_size <= 0:
            raise ValueError(f"[{branch_name}] {role_name}.ulysses_size must be positive, got {ulysses_size}.")
        if world_size % ulysses_size != 0:
            raise ValueError(
                f"[{branch_name}] {role_name} world_size={world_size} is not divisible by "
                f"ulysses_size={ulysses_size}."
            )

        effective_global_batch_size = role_config.global_batch_size
        if rollout_n > 1:
            effective_global_batch_size *= rollout_n
        global_batch_size_per_device = effective_global_batch_size // (world_size // ulysses_size)
        if global_batch_size_per_device == 0:
            raise ValueError(
                f"[{branch_name}] {role_name} global batch size * ulysses size must be larger than num gpus. "
                f"Got effective_global_batch_size={effective_global_batch_size}, world_size={world_size}, "
                f"ulysses_size={ulysses_size}."
            )

        micro_batch_size = role_config.micro_batch_size_per_device_for_update
        if micro_batch_size <= 0:
            raise ValueError(
                f"[{branch_name}] {role_name}.micro_batch_size_per_device_for_update must be positive, "
                f"got {micro_batch_size}."
            )
        if global_batch_size_per_device % micro_batch_size != 0:
            raise ValueError(
                f"[{branch_name}] {role_name} global batch size per device={global_batch_size_per_device} "
                f"is not divisible by micro batch size={micro_batch_size}."
            )

        if role_config.fsdp.enable_cpu_offload and global_batch_size_per_device != micro_batch_size:
            raise ValueError(
                f"[{branch_name}] {role_name} cannot use FSDP CPU offload when gradient accumulation is enabled. "
                f"Got global_batch_size_per_device={global_batch_size_per_device}, "
                f"micro_batch_size_per_device_for_update={micro_batch_size}."
            )

    def _validate_branch_distribution(self) -> None:
        if self.node_allocation is not None:
            for spec, branch_config in self.branch_prepared:
                proc = self.node_allocation[spec.name]
                world_size = sum(proc)
                tp_size = branch_config.worker.rollout.tensor_parallel_size
                if tp_size <= 0:
                    raise ValueError(
                        f"[{spec.name}] worker.rollout.tensor_parallel_size must be positive, got {tp_size}."
                    )
                if world_size % tp_size != 0:
                    raise ValueError(
                        f"[{spec.name}] rollout world size {world_size} is not divisible by "
                        f"tensor_parallel_size {tp_size}."
                    )
                self._validate_role_batch_geometry(
                    role_name="actor",
                    role_config=branch_config.worker.actor,
                    rollout_n=branch_config.worker.rollout.n,
                    world_size=world_size,
                    branch_name=spec.name,
                )
                if self.use_critic:
                    self._validate_role_batch_geometry(
                        role_name="critic",
                        role_config=branch_config.worker.critic,
                        rollout_n=branch_config.worker.rollout.n,
                        world_size=world_size,
                        branch_name=spec.name,
                    )
            return

        total_gpus_per_node = sum(self.branch_gpu_counts.values())
        if total_gpus_per_node > self.config.trainer.n_gpus_per_node:
            raise ValueError(
                f"Tri-branch allocation requires {total_gpus_per_node} GPUs per node, "
                f"but trainer.n_gpus_per_node={self.config.trainer.n_gpus_per_node}."
            )

        for spec, branch_config in self.branch_prepared:
            gpus_per_model = self.branch_gpu_counts[spec.name]
            world_size = gpus_per_model * self.config.trainer.nnodes
            tp_size = branch_config.worker.rollout.tensor_parallel_size
            if tp_size <= 0:
                raise ValueError(
                    f"[{spec.name}] worker.rollout.tensor_parallel_size must be positive, got {tp_size}."
                )
            if world_size % tp_size != 0:
                raise ValueError(
                    f"[{spec.name}] rollout world size {world_size} is not divisible by "
                    f"tensor_parallel_size {tp_size}."
                )

            self._validate_role_batch_geometry(
                role_name="actor",
                role_config=branch_config.worker.actor,
                rollout_n=branch_config.worker.rollout.n,
                world_size=world_size,
                branch_name=spec.name,
            )
            if self.use_critic:
                self._validate_role_batch_geometry(
                    role_name="critic",
                    role_config=branch_config.worker.critic,
                    rollout_n=branch_config.worker.rollout.n,
                    world_size=world_size,
                    branch_name=spec.name,
                )

    def _apply_branch_overrides(self, branch_config: PPOConfig, spec: TriBranchSpec, index: int) -> None:
        base_seed = self.config.worker.rollout.seed
        branch_config.worker.rollout.seed = spec.rollout_seed if spec.rollout_seed is not None else base_seed + index

        if spec.rollout_temperature is not None:
            branch_config.worker.rollout.temperature = spec.rollout_temperature
        if spec.rollout_top_p is not None:
            branch_config.worker.rollout.top_p = spec.rollout_top_p
        if spec.rollout_batch_size is not None:
            branch_config.data.rollout_batch_size = spec.rollout_batch_size
        if spec.actor_global_batch_size is not None:
            branch_config.worker.actor.global_batch_size = spec.actor_global_batch_size
        if spec.critic_global_batch_size is not None:
            branch_config.worker.critic.global_batch_size = spec.critic_global_batch_size
        if spec.model_path is not None:
            branch_config.worker.actor.model.model_path = spec.model_path

        branch_config.worker.actor.use_kl_loss = False

    def _prepare_branch_optim_steps(self, branch_config: PPOConfig) -> None:
        branch_config.worker.actor.optim.training_steps = self.training_steps
        branch_config.worker.critic.optim.training_steps = self.training_steps

    def init_workers(self):
        gpus_available = ray.available_resources().get("GPU", 0)
        if self.node_allocation is not None:
            gpus_required = sum(sum(proc) for proc in self.node_allocation.values())
        else:
            gpus_required = sum(self.branch_gpu_counts.values()) * self.config.trainer.nnodes
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than required GPUs {gpus_required}.")

        for spec, branch_config in self.branch_prepared:
            if self.node_allocation is not None:
                process_on_nodes = self.node_allocation[spec.name]
            else:
                gpus_per_model = self.branch_gpu_counts[spec.name]
                process_on_nodes = [gpus_per_model] * self.config.trainer.nnodes
            pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=f"coev_{spec.name}",
            )
            n_active_nodes = sum(1 for x in process_on_nodes if x > 0)
            print(
                f"[init_workers] Initializing {spec.name}: {process_on_nodes} "
                f"({n_active_nodes} nodes, {sum(process_on_nodes)} GPUs total)...",
                flush=True,
            )
            branch = self._init_branch(spec, pool, branch_config)
            self.branches.append(branch)
            self.branch_map[branch.name] = branch
            print(f"[init_workers] {spec.name} done.", flush=True)

    def _init_branch(
        self,
        spec: TriBranchSpec,
        resource_pool: RayResourcePool,
        branch_config: PPOConfig,
    ) -> TriBranchState:
        t0 = time.time()
        print(f"  [{spec.name}] Creating class_dict...", flush=True)
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

        print(f"  [{spec.name}] Creating worker_dict_cls...", flush=True)
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        print(f"  [{spec.name}] Creating worker group...", flush=True)
        wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        print(f"  [{spec.name}] Spawning workers...", flush=True)
        spawned = wg_dict.spawn(prefix_set=class_dict.keys())
        print(f"  [{spec.name}] Workers spawned in {time.time() - t0:.1f}s", flush=True)
        self.wg_dicts.append(wg_dict)

        critic_wg = spawned.get("critic")
        if critic_wg is not None:
            print(f"  [{spec.name}] Calling critic.init_model()...", flush=True)
            critic_wg.init_model()
            print(f"  [{spec.name}] critic.init_model() done in {time.time() - t0:.1f}s", flush=True)
        actor_rollout_ref_wg = spawned["actor_rollout_ref"]
        print(f"  [{spec.name}] Calling actor_rollout_ref.init_model()...", flush=True)
        actor_rollout_ref_wg.init_model()
        print(f"  [{spec.name}] actor_rollout_ref.init_model() done in {time.time() - t0:.1f}s", flush=True)

        kl_ctrl = (
            get_kl_controller(branch_config.algorithm)
            if not branch_config.algorithm.disable_kl
            else FixedKLController(init_kl_coef=0.0)
        )
        return TriBranchState(
            name=spec.name,
            config=branch_config,
            train_dataloader=spec.train_dataloader,
            val_dataloader=spec.val_dataloader,
            actor_rollout_ref_wg=actor_rollout_ref_wg,
            critic_wg=critic_wg,
            kl_ctrl=kl_ctrl,
            reward_fn=spec.reward_fn or self.reward_fn,
            val_reward_fn=spec.val_reward_fn or self.val_reward_fn,
            reverse_kl_coef=spec.reverse_kl_coef,
        )

    def _save_branch_checkpoint(self, branch: TriBranchState, step: int):
        dir_format = f"{branch.name}_step_{{}}"
        if self.branch_val_reward_scores[branch.name] > self.best_branch_val_reward_scores[branch.name]:
            self.best_branch_val_reward_scores[branch.name] = self.branch_val_reward_scores[branch.name]
            self.best_branch_steps[branch.name] = step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            step,
            self.best_branch_steps[branch.name] or 0,
            self.config.trainer.save_limit,
            directory_format=dir_format,
        )

        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, dir_format.format(step))
        actor_path = os.path.join(folder_path, f"actor_{branch.name}")
        branch.actor_rollout_ref_wg.save_checkpoint(
            actor_path, save_model_only=self.config.trainer.save_model_only
        )
        if self.use_critic and branch.critic_wg is not None:
            critic_path = os.path.join(folder_path, f"critic_{branch.name}")
            branch.critic_wg.save_checkpoint(
                critic_path, save_model_only=self.config.trainer.save_model_only
            )
        torch.save(branch.train_dataloader.state_dict(), os.path.join(folder_path, f"{branch.name}_dataloader.pt"))

        with self._tracker_lock:
            tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
            tracker_info = {
                "global_step": self.global_step,
                "branch_steps": dict(self.branch_steps),
                "best_steps": dict(self.best_branch_steps),
                "best_val_reward_scores": {
                    name: round(score, 4) for name, score in self.best_branch_val_reward_scores.items()
                },
                "last_ckpt_dirs": {},
            }
            if os.path.exists(tracker_path):
                with open(tracker_path, "r") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    tracker_info.update(existing)
                    tracker_info["global_step"] = self.global_step
                    tracker_info["branch_steps"] = dict(self.branch_steps)
                    tracker_info["best_steps"] = dict(self.best_branch_steps)
                    tracker_info["best_val_reward_scores"] = {
                        name: round(score, 4) for name, score in self.best_branch_val_reward_scores.items()
                    }
                    tracker_info.setdefault("last_ckpt_dirs", {})

            tracker_info["last_ckpt_dirs"][branch.name] = dir_format.format(step)
            with open(tracker_path, "w") as f:
                json.dump(tracker_info, f, ensure_ascii=False, indent=2)

    def _save_checkpoint(self):
        for branch in self.branches:
            self._save_branch_checkpoint(branch, self.branch_steps[branch.name])

    def _normalize_checkpoint_root(self, checkpoint_path: Optional[str]) -> Optional[str]:
        if checkpoint_path is None:
            return None
        if os.path.isfile(checkpoint_path):
            if os.path.basename(checkpoint_path) != CHECKPOINT_TRACKER:
                raise ValueError(
                    "Tri-branch `load_checkpoint_path` must point to a checkpoint root directory "
                    f"or `{CHECKPOINT_TRACKER}`, got file: {checkpoint_path}"
                )
            return os.path.dirname(checkpoint_path)
        return checkpoint_path

    def _find_latest_ckpt(self, checkpoint_path: Optional[str] = None) -> tuple[Optional[str], Optional[dict[str, Any]]]:
        save_path = self._normalize_checkpoint_root(
            checkpoint_path if checkpoint_path is not None else self.config.trainer.save_checkpoint_path
        )
        if save_path is None:
            return None, None
        tracker_path = os.path.join(save_path, CHECKPOINT_TRACKER)
        if not os.path.exists(tracker_path):
            return None, None

        with open(tracker_path, "r") as f:
            tracker_info = json.load(f)

        last_ckpt_dirs = tracker_info.get("last_ckpt_dirs", {})
        if not isinstance(last_ckpt_dirs, dict):
            return None, None

        for branch in self.branches:
            ckpt_dir = last_ckpt_dirs.get(branch.name)
            if ckpt_dir is None or not os.path.exists(os.path.join(save_path, ckpt_dir)):
                return None, None
        return save_path, tracker_info

    def _restore_from_tracker_checkpoint(self, save_path: str, tracker_info: dict[str, Any]) -> None:
        self.global_step = tracker_info.get("global_step", 0)
        self.branch_steps.update(tracker_info.get("branch_steps", {}))
        self.best_branch_steps.update(tracker_info.get("best_steps", {}))
        self.best_branch_val_reward_scores.update(tracker_info.get("best_val_reward_scores", {}))
        last_ckpt_dirs = tracker_info.get("last_ckpt_dirs", {})

        for branch in self.branches:
            ckpt_path = os.path.join(save_path, last_ckpt_dirs[branch.name])
            branch.actor_rollout_ref_wg.load_checkpoint(os.path.join(ckpt_path, f"actor_{branch.name}"))
            if self.use_critic and branch.critic_wg is not None:
                branch.critic_wg.load_checkpoint(os.path.join(ckpt_path, f"critic_{branch.name}"))

            dl_path = os.path.join(ckpt_path, f"{branch.name}_dataloader.pt")
            if os.path.exists(dl_path):
                branch.train_dataloader.load_state_dict(torch.load(dl_path, weights_only=False))

    def _load_checkpoint(self):
        load_checkpoint_path = self.config.trainer.load_checkpoint_path
        tracker_info = None
        save_path = None
        if load_checkpoint_path is not None:
            save_path, tracker_info = self._find_latest_ckpt(load_checkpoint_path)
            if tracker_info is None:
                raise ValueError(
                    "Tri-branch `trainer.load_checkpoint_path` must point to a checkpoint root directory "
                    f"or `{CHECKPOINT_TRACKER}` containing valid `last_ckpt_dirs` for all branches. "
                    f"Got: {load_checkpoint_path}"
                )
            print(f"[tri-branch] Loading checkpoint from explicit path: {save_path}", flush=True)
        elif self.config.trainer.find_last_checkpoint:
            save_path, tracker_info = self._find_latest_ckpt()
        if tracker_info is None:
            return

        self._restore_from_tracker_checkpoint(save_path, tracker_info)

        print(
            f"[tri-branch] Resumed from tracker checkpoint at global_step={self.global_step}.",
            flush=True,
        )

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

    def _apply_online_filtering(
        self,
        batch: DataProto,
        metrics_store: defaultdict[str, list],
        reward_fn,
    ) -> DataProto:
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
        common_keys = set(batches[0].non_tensor_batch.keys())
        for batch in batches[1:]:
            common_keys &= set(batch.non_tensor_batch.keys())
        for batch in batches:
            for key in list(batch.non_tensor_batch.keys()):
                if key not in common_keys:
                    del batch.non_tensor_batch[key]

        for key in common_keys:
            shapes = [b.non_tensor_batch[key].shape[1:] for b in batches]
            ndims = [b.non_tensor_batch[key].ndim for b in batches]
            if len(set(ndims)) > 1 or len(set(shapes)) > 1:
                for b in batches:
                    arr = b.non_tensor_batch[key]
                    flat = np.empty(arr.shape[0], dtype=object)
                    for i in range(arr.shape[0]):
                        flat[i] = arr[i]
                    b.non_tensor_batch[key] = flat

    def _is_opd_active(self, step: int) -> bool:
        off = self.config.coev.opd_off_steps
        on = self.config.coev.opd_on_steps
        if off <= 0 and on <= 0:
            return self.config.coev.cross_reward_loss_enabled
        cycle = off + on
        if cycle <= 0:
            return self.config.coev.cross_reward_loss_enabled
        return ((step - 1) % cycle) >= off

    def _is_final_opd(self, step: int) -> bool:
        return self.config.coev.force_final_opd


    def _get_reverse_kl_coef(self, branch: TriBranchState) -> float:
        if branch.reverse_kl_coef is not None:
            return branch.reverse_kl_coef
        return self.config.coev.reverse_kl_coef

    def _is_branch_frozen(self, branch: TriBranchState) -> bool:
        if not self.config.coev.freeze_zero_opd_branches:
            return False
        return float(self._get_reverse_kl_coef(branch)) == 0.0

    def _trainable_branches(self) -> list[TriBranchState]:
        return [branch for branch in self.branches if not self._is_branch_frozen(branch)]

    def _compute_cross_reward(
        self,
        scorer_branch: TriBranchState,
        target_batch: DataProto,
        target_name: str,
        metrics: dict[str, Any],
    ) -> None:
        cross_batch = target_batch.select(
            batch_keys=["input_ids", "attention_mask", "position_ids", "responses", "response_mask"],
            non_tensor_batch_keys=list(target_batch.non_tensor_batch.keys()),
        )
        cross_log_probs = scorer_branch.actor_rollout_ref_wg.compute_log_probs(cross_batch)
        log_p = cross_log_probs.batch["old_log_probs"]
        mask = target_batch.batch["response_mask"]
        seq_scores = (log_p * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

        if self.config.coev.cross_reward_style == "reward_bonus":
            bonus = self.config.coev.anchor_coef * seq_scores.unsqueeze(-1) * mask
            target_batch.batch["token_level_scores"] = target_batch.batch["token_level_scores"] + bonus
        else:
            target_batch.batch["peer_log_probs"] = log_p

        metrics[f"{target_name}/coev/cross_reward_applied"] = 1.0
        metrics[f"{target_name}/coev/cross_reward_mean"] = seq_scores.mean().item()
        metrics[f"{target_name}/coev/cross_reward_std"] = seq_scores.std().item() if seq_scores.numel() > 1 else 0.0

    def _prepare_batch(
        self,
        branch: TriBranchState,
        batch: DataProto,
        metrics: dict[str, Any],
        scorer_branch: Optional[TriBranchState] = None,
        metric_prefix: Optional[str] = None,
        is_final_opd: bool = False,
    ) -> DataProto:
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

        if "peer_log_probs" in batch.batch:
            reverse_kl_coef = self._get_reverse_kl_coef(branch)
            old_log_probs = batch.batch["old_log_probs"]
            peer_log_probs = batch.batch["peer_log_probs"]
            mask = batch.batch["response_mask"]
            reverse_kl = old_log_probs - peer_log_probs
            if is_final_opd:
                batch.batch["advantages"] = reverse_kl_coef * torch.clamp(-reverse_kl, min=0.0) * mask
            else:
                if self.config.coev.cross_reward_style == "reverse_kl_positive":
                    kl_advantage = torch.clamp(-reverse_kl, min=0.0) * mask
                else:
                    kl_advantage = -reverse_kl * mask
                batch.batch["advantages"] = (
                    (1 - reverse_kl_coef) * batch.batch["advantages"] + reverse_kl_coef * kl_advantage
                )
            seq_kl = (reverse_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            metrics[f"{prefix}/coev/reverse_kl_mean"] = seq_kl.mean().item()
            metrics[f"{prefix}/coev/reverse_kl_std"] = seq_kl.std().item() if seq_kl.numel() > 1 else 0.0
            del batch.batch["peer_log_probs"]

        return batch

    def _foreign_update_has_signal(self, branch: TriBranchState) -> bool:
        if self.config.coev.foreign_grpo_loss_enabled:
            return True

        if self.config.coev.cross_reward_style == "reward_bonus":
            return self.config.coev.anchor_coef != 0.0

        return self._get_reverse_kl_coef(branch) != 0.0

    def _update_branch(
        self,
        branch: TriBranchState,
        batch: Optional[DataProto],
        metrics: dict[str, Any],
        step: int,
        metric_prefix: Optional[str] = None,
    ) -> None:
        prefix = metric_prefix if metric_prefix is not None else branch.name
        if self._is_branch_frozen(branch):
            metrics[f"{prefix}/coev/frozen_zero_opd"] = 1.0
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

    def _validate_branch(self, branch: TriBranchState, dataloader: StatefulDataLoader, step: int = 0) -> dict[str, Any]:
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
            group_type_lst = data_type_lst
            if valid_data_type and len(set(data_type_lst)) == 1 and valid_problem_type:
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

    def _rollout_from_batch_dict(
        self,
        branch: TriBranchState,
        batch_dict: dict,
        metrics: dict[str, Any],
    ) -> DataProto:
        all_metrics = defaultdict(list)
        new_batch = DataProto.from_single_dict(batch_dict, meta_info=self._build_meta_info())
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object,
        )

        gen_batch = self._make_generation_batch(new_batch)
        branch.actor_rollout_ref_wg.prepare_rollout_engine()
        gen_output = branch.actor_rollout_ref_wg.generate_sequences(gen_batch)
        branch.actor_rollout_ref_wg.release_rollout_engine()

        new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
        new_batch = new_batch.union(gen_output)

        if self.config.algorithm.online_filtering:
            new_batch = self._apply_online_filtering(new_batch, all_metrics, branch.reward_fn)
            metrics.update(
                _prefix_metrics(branch.name, {f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
            )

        rollout_batch_size = branch.config.data.rollout_batch_size
        return new_batch[: rollout_batch_size * self.config.worker.rollout.n]

    def _rollout_branch_merged(
        self,
        branch: TriBranchState,
        source_batch_dicts: dict[str, dict],
        metrics: dict[str, Any],
        timing: dict[str, float],
    ) -> dict[str, DataProto]:
        with timer("gen", timing):
            source_protos: list[DataProto] = []
            source_names = []
            source_sizes = []
            for source_name, batch_dict in source_batch_dicts.items():
                proto = DataProto.from_single_dict(batch_dict, meta_info=self._build_meta_info())
                source_protos.append(proto)
                source_names.append(source_name)
                source_sizes.append(len(proto))
            self._align_non_tensor_keys(*source_protos)

            merged = DataProto.concat(source_protos)
            merged.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(merged.batch))], dtype=object,
            )
            merged.non_tensor_batch["_source_branch"] = np.concatenate(
                [np.array([source_name] * size, dtype=object) for source_name, size in zip(source_names, source_sizes)],
                axis=0,
            )

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
            split_batches = {}
            source_branch = merged.non_tensor_batch["_source_branch"]
            for source_name in source_names:
                source_idxs = np.where(source_branch == source_name)[0][: rollout_batch_size * n]
                split_batch = merged[source_idxs]
                del split_batch.non_tensor_batch["_source_branch"]
                split_batches[source_name] = split_batch

        return split_batches

    def _setup_wandb_metrics(self):
        try:
            pass
        except Exception:
            pass

    def fit(self):
        t_fit = time.time()
        print(f"[tri-branch fit] Creating logger...", flush=True)
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        print(f"[tri-branch fit] Logger created in {time.time() - t_fit:.1f}s", flush=True)
        self._setup_wandb_metrics()

        t_ckpt = time.time()
        print(f"[tri-branch fit] Loading checkpoint...", flush=True)
        self._load_checkpoint()
        print(f"[tri-branch fit] Checkpoint load finished in {time.time() - t_ckpt:.1f}s", flush=True)
        if self.hub_branch_name is None:
            print("[tri-branch fit] Using full-graph topology.", flush=True)
        else:
            print(f"[tri-branch fit] Using hub topology with hub={self.hub_branch_name}.", flush=True)
        if self.config.coev.freeze_zero_opd_branches:
            frozen_names = [branch.name for branch in self.branches if self._is_branch_frozen(branch)]
            print(
                "[tri-branch fit] freeze_zero_opd_branches enabled; frozen branches: "
                + (", ".join(frozen_names) if frozen_names else "none"),
                flush=True,
            )

        val_enabled_branches = [branch for branch in self.branches if branch.val_reward_fn is not None]
        if val_enabled_branches and self.config.trainer.val_before_train:
            t_val = time.time()
            print(f"[tri-branch fit] val_before_train: starting parallel validation...", flush=True)
            val_metrics: dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=len(val_enabled_branches)) as executor:
                future_map = {
                    executor.submit(self._validate_branch, branch, branch.val_dataloader, step=self.branch_steps[branch.name]): branch
                    for branch in val_enabled_branches
                }
                for future, branch in future_map.items():
                    branch_metrics = future.result()
                    val_metrics.update(branch_metrics)
                    self.branch_val_reward_scores[branch.name] = branch_metrics.get(f"{branch.name}/val/reward_score", 0.0)
            print(f"[tri-branch fit] val_before_train done in {time.time() - t_val:.1f}s", flush=True)

            combined_val = dict(val_metrics)
            combined_val["train/global_step"] = self.global_step
            for branch in self.branches:
                combined_val[f"{branch.name}_step"] = self.branch_steps[branch.name]
            self.logger.log(data=combined_val)
            if self.config.trainer.val_only:
                return

        print("[tri-branch fit] Starting cross_reward training loop...", flush=True)
        self._fit_cross_reward()
        self._save_checkpoint()

    def _fit_cross_reward(self):
        val_freq = self.config.trainer.val_freq
        save_freq = self.config.trainer.save_freq
        iterator_attrs = {branch.name: f"{branch.name}_data_iterator" for branch in self.branches}
        for branch in self.branches:
            setattr(self, iterator_attrs[branch.name], iter(branch.train_dataloader))

        while self.global_step < self.training_steps:
            step = self.global_step + 1
            self.global_step = step
            for branch in self.branches:
                self.branch_steps[branch.name] = step

            metrics_by_branch = {branch.name: {} for branch in self.branches}
            timing_by_branch = {branch.name: {} for branch in self.branches}
            trainable_branches = self._trainable_branches()
            trainable_branch_names = {branch.name for branch in trainable_branches}
            for branch in self.branches:
                is_frozen = self._is_branch_frozen(branch)
                metrics_by_branch[branch.name][f"{branch.name}/coev/frozen_zero_opd"] = float(is_frozen)
                if is_frozen:
                    metrics_by_branch[branch.name][f"{branch.name}/coev/update_skipped"] = 1.0

            final_opd_active = self._is_final_opd(step)
            opd_active = self._is_opd_active(step) or final_opd_active
            needed_batch_names = set()
            if opd_active:
                for branch in trainable_branches:
                    needed_batch_names.update(self._get_rollout_source_names(branch.name))
            else:
                needed_batch_names.update(trainable_branch_names)
            native_batch_dicts = {
                branch.name: self._next_batch_dict(iterator_attrs[branch.name], branch.train_dataloader)
                for branch in self.branches
                if branch.name in needed_batch_names
            }

            for branch in self.branches:
                for source_name in self._get_connected_foreign_branch_names(branch.name):
                    prefix = f"{branch.name}/foreign/{source_name}"
                    metrics_by_branch[branch.name][f"{prefix}/coev/cross_reward_applied"] = 0.0
                    metrics_by_branch[branch.name][f"{prefix}/coev/cross_reward_mean"] = 0.0
                    metrics_by_branch[branch.name][f"{prefix}/coev/cross_reward_std"] = 0.0

            if opd_active:
                rollout_splits: dict[str, dict[str, DataProto]] = {}
                if trainable_branches:
                    with ThreadPoolExecutor(max_workers=len(trainable_branches)) as executor:
                        future_map = {
                            executor.submit(
                                self._rollout_branch_merged,
                                branch,
                                {
                                    source_name: native_batch_dicts[source_name]
                                    for source_name in self._get_rollout_source_names(branch.name)
                                },
                                metrics_by_branch[branch.name],
                                timing_by_branch[branch.name],
                            ): branch.name
                            for branch in trainable_branches
                        }
                        for future, branch_name in future_map.items():
                            rollout_splits[branch_name] = future.result()

                for branch in trainable_branches:
                    for source_name, split_batch in rollout_splits[branch.name].items():
                        prefix = branch.name if branch.name == source_name else f"{branch.name}/foreign/{source_name}"
                        self._balance_batch(
                            split_batch,
                            branch.actor_rollout_ref_wg.world_size,
                            metrics_by_branch[branch.name],
                            logging_prefix=prefix,
                        )
                        split_batch.meta_info["global_token_num"] = torch.sum(
                            split_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                prepared_native = {}
                if trainable_branches:
                    with ThreadPoolExecutor(max_workers=len(trainable_branches)) as executor:
                        future_map = {
                            executor.submit(
                                self._prepare_batch,
                                branch,
                                rollout_splits[branch.name][branch.name],
                                metrics_by_branch[branch.name],
                                is_final_opd=final_opd_active,
                            ): branch.name
                            for branch in trainable_branches
                        }
                        for future, branch_name in future_map.items():
                            prepared_native[branch_name] = future.result()

                prepared_foreign: dict[str, list[DataProto]] = {branch.name: [] for branch in trainable_branches}
                for branch in trainable_branches:
                    for source_name in self._get_connected_foreign_branch_names(branch.name):
                        source = self.branch_map[source_name]
                        prefix = f"{branch.name}/foreign/{source.name}"
                        foreign_batch = self._prepare_batch(
                            branch,
                            rollout_splits[branch.name][source.name],
                            metrics_by_branch[branch.name],
                            scorer_branch=source,
                            metric_prefix=prefix,
                            is_final_opd=final_opd_active,
                        )
                        prepared_foreign[branch.name].append(foreign_batch)

                combined_batches = {branch.name: None for branch in self.branches}
                for branch in trainable_branches:
                    foreign_enabled = self._foreign_update_has_signal(branch)
                    metrics_by_branch[branch.name][f"{branch.name}/foreign/coev/update_skipped"] = (
                        0.0 if foreign_enabled else 1.0
                    )
                    if final_opd_active:
                        all_parts = prepared_foreign[branch.name] if foreign_enabled else []
                    else:
                        all_parts = (
                            [prepared_native[branch.name], *prepared_foreign[branch.name]]
                            if foreign_enabled
                            else [prepared_native[branch.name]]
                        )

                    if len(all_parts) == 0:
                        combined = None
                    elif len(all_parts) == 1:
                        combined = all_parts[0]
                    else:
                        self._align_non_tensor_keys(*all_parts)
                        combined = DataProto.concat(all_parts)
                        combined.meta_info["global_token_num"] = []
                        for part in all_parts:
                            combined.meta_info["global_token_num"].extend(part.meta_info.get("global_token_num", []))
                    combined_batches[branch.name] = combined
                    rollout_splits[branch.name][branch.name] = prepared_native[branch.name]
            else:
                native_batches = {}
                if trainable_branches:
                    with ThreadPoolExecutor(max_workers=len(trainable_branches)) as executor:
                        future_map = {
                            executor.submit(
                                self._rollout_from_batch_dict,
                                branch,
                                native_batch_dicts[branch.name],
                                metrics_by_branch[branch.name],
                            ): branch.name
                            for branch in trainable_branches
                        }
                        for future, branch_name in future_map.items():
                            native_batches[branch_name] = future.result()

                for branch in trainable_branches:
                    self._balance_batch(
                        native_batches[branch.name],
                        branch.actor_rollout_ref_wg.world_size,
                        metrics_by_branch[branch.name],
                        logging_prefix=branch.name,
                    )
                    native_batches[branch.name].meta_info["global_token_num"] = torch.sum(
                        native_batches[branch.name].batch["attention_mask"], dim=-1
                    ).tolist()

                combined_batches = {branch.name: None for branch in self.branches}
                if trainable_branches:
                    with ThreadPoolExecutor(max_workers=len(trainable_branches)) as executor:
                        future_map = {
                            executor.submit(
                                self._prepare_batch,
                                branch,
                                native_batches[branch.name],
                                metrics_by_branch[branch.name],
                                is_final_opd=final_opd_active,
                            ): branch.name
                            for branch in trainable_branches
                        }
                        for future, branch_name in future_map.items():
                            combined_batches[branch_name] = future.result()
                rollout_splits = {
                    branch.name: {branch.name: combined_batches[branch.name]}
                    for branch in trainable_branches
                }

            with ThreadPoolExecutor(max_workers=self.branch_count) as executor:
                future_map = {
                    executor.submit(
                        self._update_branch,
                        branch,
                        combined_batches[branch.name],
                        metrics_by_branch[branch.name],
                        step,
                    ): branch.name
                    for branch in self.branches
                }
                for future in future_map:
                    future.result()

            val_enabled_branches = [branch for branch in self.branches if branch.val_reward_fn is not None]
            if val_enabled_branches and val_freq > 0 and step % val_freq == 0:
                with ThreadPoolExecutor(max_workers=len(val_enabled_branches)) as executor:
                    future_map = {
                        executor.submit(self._validate_branch, branch, branch.val_dataloader, step=step): branch.name
                        for branch in val_enabled_branches
                    }
                    for future, branch_name in future_map.items():
                        branch_metrics = future.result()
                        metrics_by_branch[branch_name].update(branch_metrics)
                        self.branch_val_reward_scores[branch_name] = branch_metrics.get(
                            f"{branch_name}/val/reward_score", 0.0
                        )

            if save_freq > 0 and step % save_freq == 0:
                with ThreadPoolExecutor(max_workers=self.branch_count) as executor:
                    future_map = {
                        executor.submit(self._save_branch_checkpoint, branch, step): branch.name
                        for branch in self.branches
                    }
                    for future in future_map:
                        future.result()

            combined_metrics: dict[str, Any] = {
                "train/global_step": step,
                "coev/anchor_coef": self.config.coev.anchor_coef,
                "coev/foreign_grpo_loss_enabled": float(self.config.coev.foreign_grpo_loss_enabled),
                "coev/cross_reward_loss_enabled": float(self.config.coev.cross_reward_loss_enabled),
                "coev/opd_active": float(opd_active),
                "coev/force_final_opd": float(self.config.coev.force_final_opd),
                "coev/hub_topology_enabled": float(self.hub_branch_name is not None),
                "coev/freeze_zero_opd_branches": float(self.config.coev.freeze_zero_opd_branches),
            }
            for branch in self.branches:
                native_batch = rollout_splits.get(branch.name, {}).get(branch.name)
                metrics = metrics_by_branch[branch.name]
                timing = timing_by_branch[branch.name]
                if native_batch is not None:
                    metrics.update(
                        _prefix_metrics(branch.name, compute_data_metrics(batch=native_batch, use_critic=self.use_critic))
                    )
                else:
                    metrics[f"{branch.name}/coev/rollout_skipped"] = 1.0
                metrics.update({f"{branch.name}/timing_s/{key}": value for key, value in timing.items()})
                metrics[f"{branch.name}/coev/reverse_kl_coef"] = self._get_reverse_kl_coef(branch)
                metrics.setdefault(f"{branch.name}/coev/update_skipped", 0.0)
                metrics[f"{branch.name}_step"] = step
                combined_metrics.update(metrics)

            self.logger.log(data=combined_metrics)
            print(
                f"[tri-branch cross_reward] step {step}/{self.training_steps} done. "
                f"(opd={'ON' if opd_active else 'OFF'})",
                flush=True,
            )
