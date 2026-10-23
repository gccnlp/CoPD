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
Dual-model PPO trainer.

This trainer keeps two actor-rollout-ref branches alive at the same time, generates
rollouts on the same prompts, injects peer responses into the tail of each branch's
sample group, and then updates both models independently.
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, compute_advantage_return, get_kl_controller
from .metrics import compute_data_metrics, compute_length_metrics, reduce_metrics
from .ray_trainer import apply_kl_penalty


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


@dataclass
class BranchState:
    name: str
    config: PPOConfig
    actor_rollout_ref_wg: RayWorkerGroup
    critic_wg: Optional[RayWorkerGroup]
    kl_ctrl: Any


class DualRayPPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.ray_worker_group_cls = ray_worker_group_cls
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        if not self.config.parallel.enabled:
            raise ValueError("DualRayPPOTrainer requires `parallel.enabled=true`.")
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("DualRayPPOTrainer does not support REMAX yet.")

        self.use_reference_policy = not self.config.algorithm.disable_kl
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
            num_examples = len(train_dataloader) * self.config.data.mini_rollout_batch_size
            self.training_steps = num_examples // self.config.data.rollout_batch_size * self.config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * self.config.trainer.total_epochs

        self.peer_replacement_count = max(1, int(self.config.parallel.peer_replacement_count))
        self.num_gpus_per_model = self.config.parallel.n_gpus_per_model
        if self.num_gpus_per_model is None:
            self.num_gpus_per_model = self.config.trainer.n_gpus_per_node // 2
        if self.num_gpus_per_model <= 0:
            raise ValueError("`parallel.n_gpus_per_model` must be positive.")
        if self.num_gpus_per_model * 2 > self.config.trainer.n_gpus_per_node:
            raise ValueError("Two parallel models require `2 * n_gpus_per_model <= trainer.n_gpus_per_node`.")

        self.branch_config_a = deepcopy(self.config)
        self.branch_config_b = deepcopy(self.config)
        self._apply_branch_overrides()
        self._prepare_branch_optim_steps(self.branch_config_a)
        self._prepare_branch_optim_steps(self.branch_config_b)

        self.global_step = 0
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.wg_dicts = []

    def _apply_branch_overrides(self):
        seed_a = self.config.parallel.rollout_seed_a
        seed_b = self.config.parallel.rollout_seed_b
        if seed_a is None:
            seed_a = self.config.worker.rollout.seed
        if seed_b is None:
            seed_b = self.config.worker.rollout.seed + 1

        self.branch_config_a.worker.rollout.seed = seed_a
        self.branch_config_b.worker.rollout.seed = seed_b

        if self.config.parallel.rollout_temperature_a is not None:
            self.branch_config_a.worker.rollout.temperature = self.config.parallel.rollout_temperature_a
        if self.config.parallel.rollout_temperature_b is not None:
            self.branch_config_b.worker.rollout.temperature = self.config.parallel.rollout_temperature_b
        if self.config.parallel.rollout_top_p_a is not None:
            self.branch_config_a.worker.rollout.top_p = self.config.parallel.rollout_top_p_a
        if self.config.parallel.rollout_top_p_b is not None:
            self.branch_config_b.worker.rollout.top_p = self.config.parallel.rollout_top_p_b

    def _prepare_branch_optim_steps(self, branch_config: PPOConfig):
        branch_config.worker.actor.optim.training_steps = self.training_steps
        branch_config.worker.critic.optim.training_steps = self.training_steps

    def init_workers(self):
        pool_a = RayResourcePool(
            process_on_nodes=[self.num_gpus_per_model] * self.config.trainer.nnodes,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="parallel_a",
        )
        pool_b = RayResourcePool(
            process_on_nodes=[self.num_gpus_per_model] * self.config.trainer.nnodes,
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="parallel_b",
        )
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = 2 * self.num_gpus_per_model * self.config.trainer.nnodes
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than required GPUs {gpus_required}.")

        self.branch_a = self._init_branch("branch_a", pool_a, self.branch_config_a)
        self.branch_b = self._init_branch("branch_b", pool_b, self.branch_config_b)

    def _init_branch(self, branch_name: str, resource_pool: RayResourcePool, branch_config: PPOConfig) -> BranchState:
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

        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawned = wg_dict.spawn(prefix_set=class_dict.keys())
        self.wg_dicts.append(wg_dict)

        critic_wg = spawned.get("critic")
        if critic_wg is not None:
            critic_wg.init_model()
        actor_rollout_ref_wg = spawned["actor_rollout_ref"]
        actor_rollout_ref_wg.init_model()

        kl_ctrl = (
            get_kl_controller(branch_config.algorithm)
            if self.use_reference_policy
            else FixedKLController(init_kl_coef=0.0)
        )
        return BranchState(
            name=branch_name,
            config=branch_config,
            actor_rollout_ref_wg=actor_rollout_ref_wg,
            critic_wg=critic_wg,
            kl_ctrl=kl_ctrl,
        )

    def _save_checkpoint(self):
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_a_path = os.path.join(folder_path, "actor_a")
        actor_b_path = os.path.join(folder_path, "actor_b")
        self.branch_a.actor_rollout_ref_wg.save_checkpoint(
            actor_a_path, save_model_only=self.config.trainer.save_model_only
        )
        self.branch_b.actor_rollout_ref_wg.save_checkpoint(
            actor_b_path, save_model_only=self.config.trainer.save_model_only
        )

        if self.use_critic:
            critic_a_path = os.path.join(folder_path, "critic_a")
            critic_b_path = os.path.join(folder_path, "critic_b")
            self.branch_a.critic_wg.save_checkpoint(critic_a_path, save_model_only=self.config.trainer.save_model_only)
            self.branch_b.critic_wg.save_checkpoint(critic_b_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_path)

        tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_a_path": os.path.abspath(actor_a_path),
            "last_actor_b_path": os.path.abspath(actor_b_path),
        }
        tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(tracker_path, "w") as f:
            json.dump(tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self):
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])

        actor_a_path = os.path.join(load_checkpoint_path, "actor_a")
        actor_b_path = os.path.join(load_checkpoint_path, "actor_b")
        self.branch_a.actor_rollout_ref_wg.load_checkpoint(actor_a_path)
        self.branch_b.actor_rollout_ref_wg.load_checkpoint(actor_b_path)
        if self.use_critic:
            self.branch_a.critic_wg.load_checkpoint(os.path.join(load_checkpoint_path, "critic_a"))
            self.branch_b.critic_wg.load_checkpoint(os.path.join(load_checkpoint_path, "critic_b"))

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            self.train_dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

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
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=[
                "image_min_pixels",
                "image_max_pixels",
                "video_min_pixels",
                "video_max_pixels",
                "video_fps",
                "video_max_frames",
            ],
        )

    def _inject_peer_responses(self, own_output: DataProto, peer_output: DataProto) -> DataProto:
        mixed_output = deepcopy(own_output)
        n = self.config.worker.rollout.n
        replace_count = min(self.peer_replacement_count, n)
        if replace_count <= 0:
            return mixed_output

        prompt_batch_size = len(mixed_output.batch) // n
        for prompt_idx in range(prompt_batch_size):
            own_target_start = prompt_idx * n + (n - replace_count)
            peer_source_start = prompt_idx * n
            own_indices = slice(own_target_start, own_target_start + replace_count)
            peer_indices = slice(peer_source_start, peer_source_start + replace_count)
            for key in mixed_output.batch.keys():
                mixed_output.batch[key][own_indices] = peer_output.batch[key][peer_indices]
        return mixed_output

    def _apply_online_filtering(self, batch: DataProto, metrics_store: defaultdict[str, list]) -> DataProto:
        filter_reward_batch = batch.select(
            batch_keys=["responses", "response_mask"],
            non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
        )
        reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(filter_reward_batch))
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

    def _make_dual_batch_data(self, metrics: dict[str, Any]) -> tuple[DataProto, DataProto]:
        batch_a = None
        batch_b = None
        all_metrics_a = defaultdict(list)
        all_metrics_b = defaultdict(list)
        num_try_make_batch = 0
        total_gen_time = 0.0
        total_filter_time = 0.0
        print("Start generating dual batches...")

        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = self._build_meta_info()
            base_batch_a = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            base_batch_b = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            base_batch_a.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(base_batch_a.batch))], dtype=object
            )
            base_batch_b.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(base_batch_b.batch))], dtype=object
            )

            gen_batch_a = self._make_generation_batch(base_batch_a)
            gen_batch_b = self._make_generation_batch(base_batch_b)

            t_start = time.time()
            gen_output_a = self.branch_a.actor_rollout_ref_wg.generate_sequences(gen_batch_a)
            gen_output_b = self.branch_b.actor_rollout_ref_wg.generate_sequences(gen_batch_b)
            total_gen_time += time.time() - t_start

            mixed_output_a = self._inject_peer_responses(gen_output_a, gen_output_b)
            mixed_output_b = self._inject_peer_responses(gen_output_b, gen_output_a)

            base_batch_a = base_batch_a.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            base_batch_b = base_batch_b.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            base_batch_a = base_batch_a.union(mixed_output_a)
            base_batch_b = base_batch_b.union(mixed_output_b)

            if self.config.algorithm.online_filtering:
                t_start = time.time()
                base_batch_a = self._apply_online_filtering(base_batch_a, all_metrics_a)
                base_batch_b = self._apply_online_filtering(base_batch_b, all_metrics_b)
                total_filter_time += time.time() - t_start

            batch_a = DataProto.concat([batch_a, base_batch_a]) if batch_a is not None else base_batch_a
            batch_b = DataProto.concat([batch_b, base_batch_b]) if batch_b is not None else base_batch_b

            current_batch_size_a = len(batch_a) // self.config.worker.rollout.n
            current_batch_size_b = len(batch_b) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size_a >= rollout_batch_size and current_batch_size_b >= rollout_batch_size:
                if self.config.algorithm.online_filtering:
                    metrics.update(_prefix_metrics("branch_a", {f"reward/{k}": v for k, v in reduce_metrics(all_metrics_a).items()}))
                    metrics.update(_prefix_metrics("branch_b", {f"reward/{k}": v for k, v in reduce_metrics(all_metrics_b).items()}))
                    metrics["parallel/timing_s/rollout_generate_part"] = total_gen_time
                    metrics["parallel/timing_s/reward_compute_part"] = total_filter_time

                return (
                    batch_a[: self.config.data.rollout_batch_size * self.config.worker.rollout.n],
                    batch_b[: self.config.data.rollout_batch_size * self.config.worker.rollout.n],
                )

            print(
                f"{current_batch_size_a=}, {current_batch_size_b=} < {rollout_batch_size=}. Continue generating..."
            )
            max_try_make_batch = self.config.trainer.max_try_make_batch
            if max_try_make_batch > 0 and num_try_make_batch >= max_try_make_batch:
                raise RuntimeError(
                    f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                )

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

    def _optimize_branch(self, branch: BranchState, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        prefix = branch.name
        if "token_level_scores" not in batch.batch:
            reward_batch = batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(reward_batch))
            batch.batch["token_level_scores"] = reward_tensor
            metrics.update(_prefix_metrics(prefix, {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}))

        old_log_probs = branch.actor_rollout_ref_wg.compute_log_probs(batch)
        batch = batch.union(old_log_probs)

        if self.use_reference_policy:
            ref_log_probs = branch.actor_rollout_ref_wg.compute_ref_log_probs(batch)
            batch = batch.union(ref_log_probs)

        if self.use_critic:
            values = branch.critic_wg.compute_values(batch)
            batch = batch.union(values)

        if not branch.config.algorithm.use_kl_loss and self.use_reference_policy:
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

        if self.use_critic:
            critic_output = branch.critic_wg.update_critic(batch)
            metrics.update(_prefix_metrics(prefix, reduce_metrics(critic_output.non_tensor_batch)))

        if self.config.trainer.critic_warmup <= self.global_step:
            actor_output = branch.actor_rollout_ref_wg.update_actor(batch)
            metrics.update(_prefix_metrics(prefix, reduce_metrics(actor_output.non_tensor_batch)))

        return batch

    def _validate_branch(self, branch: BranchState) -> dict[str, Any]:
        reward_tensor_lst = []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []

        print(f"Start validation for {branch.name}...")
        branch.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = branch.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = branch.config.worker.rollout.val_override_config
            test_gen_batch.meta_info.update(self._build_meta_info())

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, branch.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = branch.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_batch = test_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in test_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(reward_batch))

            reward_tensor_lst.append(reward_tensor)
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
            self.logger.log_generation(samples[: self.config.trainer.val_generations_to_log], self.global_step)

        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        metrics = {
            "val/reward_score": reward_score,
            **{f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()},
            **{f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()},
        }
        print(f"Finish validation for {branch.name}: {convert_dict_to_str(metrics)}")
        return _prefix_metrics(branch.name, metrics)

    def fit(self):
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        val_metrics: Optional[dict[str, Any]] = None
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = {}
            val_metrics.update(self._validate_branch(self.branch_a))
            val_metrics.update(self._validate_branch(self.branch_b))
            reward_a = val_metrics.get("branch_a/val/reward_score", 0.0)
            reward_b = val_metrics.get("branch_b/val/reward_score", 0.0)
            self.val_reward_score = (reward_a + reward_b) / 2.0
            val_metrics["parallel/val/reward_score"] = self.val_reward_score
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1
            metrics: dict[str, Any] = {}
            timing_raw: dict[str, float] = {}

            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.branch_a.actor_rollout_ref_wg.prepare_rollout_engine()
                    self.branch_b.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch_a, batch_b = self._make_dual_batch_data(metrics=metrics)
                    self.branch_a.actor_rollout_ref_wg.release_rollout_engine()
                    self.branch_b.actor_rollout_ref_wg.release_rollout_engine()

                self._balance_batch(
                    batch_a,
                    self.branch_a.actor_rollout_ref_wg.world_size,
                    metrics,
                    logging_prefix="branch_a",
                )
                self._balance_batch(
                    batch_b,
                    self.branch_b.actor_rollout_ref_wg.world_size,
                    metrics,
                    logging_prefix="branch_b",
                )

                batch_a.meta_info["global_token_num"] = torch.sum(batch_a.batch["attention_mask"], dim=-1).tolist()
                batch_b.meta_info["global_token_num"] = torch.sum(batch_b.batch["attention_mask"], dim=-1).tolist()

                with timer("optimize_a", timing_raw):
                    batch_a = self._optimize_branch(self.branch_a, batch_a, metrics)
                with timer("optimize_b", timing_raw):
                    batch_b = self._optimize_branch(self.branch_b, batch_b, metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = {}
                        val_metrics.update(self._validate_branch(self.branch_a))
                        val_metrics.update(self._validate_branch(self.branch_b))
                        reward_a = val_metrics.get("branch_a/val/reward_score", 0.0)
                        reward_b = val_metrics.get("branch_b/val/reward_score", 0.0)
                        self.val_reward_score = (reward_a + reward_b) / 2.0
                        val_metrics["parallel/val/reward_score"] = self.val_reward_score
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            metrics.update(_prefix_metrics("branch_a", compute_data_metrics(batch=batch_a, use_critic=self.use_critic)))
            metrics.update(_prefix_metrics("branch_b", compute_data_metrics(batch=batch_b, use_critic=self.use_critic)))
            metrics.update({f"parallel/timing_s/{key}": value for key, value in timing_raw.items()})
            metrics["parallel/peer_replacement_count"] = self.peer_replacement_count
            metrics["parallel/num_gpus_total"] = 2 * self.num_gpus_per_model * self.config.trainer.nnodes

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = {}
                val_metrics.update(self._validate_branch(self.branch_a))
                val_metrics.update(self._validate_branch(self.branch_b))
                reward_a = val_metrics.get("branch_a/val/reward_score", 0.0)
                reward_b = val_metrics.get("branch_b/val/reward_score", 0.0)
                self.val_reward_score = (reward_a + reward_b) / 2.0
                val_metrics["parallel/val/reward_score"] = self.val_reward_score
                self.logger.log(data=val_metrics, step=self.global_step)

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
