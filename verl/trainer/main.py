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

import json

import ray
from omegaconf import OmegaConf
from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .coev_ray_trainer import CoEvolRayPPOTrainer
from .coev_tri_ray_trainer import TriBranchCoEvolRayPPOTrainer, TriBranchSpec
from .data_loader import create_coevol_dataloaders, create_dataloader
from .dual_ray_trainer import DualRayPPOTrainer
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_files = config.data.text_val_files if config.coev.enabled else config.data.val_files
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer) if val_files else None

        if config.coev.enabled:
            tri_branch_enabled = bool(config.data.train_files)
            (
                text_train_dataloader,
                text_val_dataloader,
                vision_train_dataloader,
                vision_val_dataloader,
            ) = create_coevol_dataloaders(
                config.data,
                tokenizer,
                processor,
                rollout_batch_size_text=(
                    config.parallel.rollout_batch_size_a
                    if config.parallel.rollout_batch_size_a is not None
                    else config.data.rollout_batch_size
                ),
                rollout_batch_size_vision=(
                    config.parallel.rollout_batch_size_b
                    if config.parallel.rollout_batch_size_b is not None
                    else config.data.rollout_batch_size
                ),
                shuffle_text=config.parallel.shuffle_a,
                shuffle_vision=config.parallel.shuffle_b,
                sequential_first_text=config.parallel.sequential_first_a,
                sequential_first_vision=config.parallel.sequential_first_b,
            )
            # Independent reward actors per branch so reward computation can run in parallel.
            reward_fn_text = RemoteRewardManager.remote(config.worker.reward, tokenizer)
            reward_fn_vision = RemoteRewardManager.remote(config.worker.reward, tokenizer)
            val_reward_fn_text = (
                RemoteRewardManager.remote(config.worker.reward, tokenizer)
                if config.data.text_val_files else None
            )
            val_reward_fn_vision = (
                RemoteRewardManager.remote(config.worker.reward, tokenizer)
                if config.data.vision_val_files else None
            )
            if tri_branch_enabled:
                video_train_dataloader, video_val_dataloader = create_dataloader(
                    config.data,
                    tokenizer,
                    processor,
                    name="video",
                    train_batch_size_override=config.parallel.rollout_batch_size_c,
                    shuffle_override=config.parallel.shuffle_c,
                    sequential_first=config.parallel.sequential_first_c,
                )
                reward_fn_video = RemoteRewardManager.remote(config.worker.reward, tokenizer)
                val_reward_fn_video = (
                    RemoteRewardManager.remote(config.worker.reward, tokenizer)
                    if config.data.val_files else None
                )
        else:
            train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        if config.coev.enabled:
            if tri_branch_enabled:
                branch_specs = [
                    TriBranchSpec(
                        name="branch_text",
                        train_dataloader=text_train_dataloader,
                        val_dataloader=text_val_dataloader,
                        reward_fn=reward_fn_text,
                        val_reward_fn=val_reward_fn_text,
                        rollout_seed=config.parallel.rollout_seed_a,
                        rollout_temperature=config.parallel.rollout_temperature_a,
                        rollout_top_p=config.parallel.rollout_top_p_a,
                        rollout_batch_size=config.parallel.rollout_batch_size_a,
                        actor_global_batch_size=config.parallel.actor_global_batch_size_a,
                        critic_global_batch_size=config.parallel.critic_global_batch_size_a,
                        model_path=config.parallel.model_path_a,
                        reverse_kl_coef=config.coev.reverse_kl_coef_text,
                    ),
                    TriBranchSpec(
                        name="branch_vision",
                        train_dataloader=vision_train_dataloader,
                        val_dataloader=vision_val_dataloader,
                        reward_fn=reward_fn_vision,
                        val_reward_fn=val_reward_fn_vision,
                        rollout_seed=config.parallel.rollout_seed_b,
                        rollout_temperature=config.parallel.rollout_temperature_b,
                        rollout_top_p=config.parallel.rollout_top_p_b,
                        rollout_batch_size=config.parallel.rollout_batch_size_b,
                        actor_global_batch_size=config.parallel.actor_global_batch_size_b,
                        critic_global_batch_size=config.parallel.critic_global_batch_size_b,
                        model_path=config.parallel.model_path_b,
                        reverse_kl_coef=config.coev.reverse_kl_coef_vision,
                    ),
                    TriBranchSpec(
                        name="branch_video",
                        train_dataloader=video_train_dataloader,
                        val_dataloader=video_val_dataloader,
                        reward_fn=reward_fn_video,
                        val_reward_fn=val_reward_fn_video,
                        rollout_seed=config.parallel.rollout_seed_c,
                        rollout_temperature=config.parallel.rollout_temperature_c,
                        rollout_top_p=config.parallel.rollout_top_p_c,
                        rollout_batch_size=config.parallel.rollout_batch_size_c,
                        actor_global_batch_size=config.parallel.actor_global_batch_size_c,
                        critic_global_batch_size=config.parallel.critic_global_batch_size_c,
                        model_path=config.parallel.model_path_c,
                        reverse_kl_coef=config.coev.reverse_kl_coef,
                    ),
                ]
                trainer = TriBranchCoEvolRayPPOTrainer(
                    config=config,
                    tokenizer=tokenizer,
                    processor=processor,
                    branch_specs=branch_specs,
                    ray_worker_group_cls=ray_worker_group_cls,
                    reward_fn=reward_fn,
                    val_reward_fn=val_reward_fn,
                )
            else:
                trainer = CoEvolRayPPOTrainer(
                    config=config,
                    tokenizer=tokenizer,
                    processor=processor,
                    text_train_dataloader=text_train_dataloader,
                    text_val_dataloader=text_val_dataloader,
                    vision_train_dataloader=vision_train_dataloader,
                    vision_val_dataloader=vision_val_dataloader,
                    ray_worker_group_cls=ray_worker_group_cls,
                    reward_fn=reward_fn,
                    val_reward_fn=val_reward_fn,
                    reward_fn_text=reward_fn_text,
                    reward_fn_vision=reward_fn_vision,
                    val_reward_fn_text=val_reward_fn_text,
                    val_reward_fn_vision=val_reward_fn_vision,
                )
            trainer.init_workers()
            trainer.fit()
            return

        if config.parallel.enabled:
            trainer = DualRayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
            )
            trainer.init_workers()
            trainer.fit()
            return

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    # Warn if per-branch model paths differ from the default — they must share the same tokenizer
    _default_model = ppo_config.worker.actor.model.model_path
    for _label, _path in [("model_path_a", ppo_config.parallel.model_path_a),
                          ("model_path_b", ppo_config.parallel.model_path_b)]:
        if _path is not None and _path != _default_model:
            print(
                f"[WARNING] parallel.{_label}={_path} differs from "
                f"worker.actor.model.model_path={_default_model}. "
                "Both checkpoints MUST share the same tokenizer/vocabulary."
            )

    wants_validation = (
        ppo_config.trainer.val_before_train
        or ppo_config.trainer.val_freq > 0
        or ppo_config.trainer.val_only
        or (ppo_config.coev.enabled and ppo_config.coev.opd_adaptive_coef)
    )
    if wants_validation:
        if ppo_config.coev.enabled:
            required = [ppo_config.data.text_val_files, ppo_config.data.vision_val_files]
            if ppo_config.data.train_files:
                required.append(ppo_config.data.val_files)
        else:
            required = [ppo_config.data.val_files]
        if not all(required):
            raise ValueError("Validation requires a validation dataset for every active branch.")

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(include_dashboard=False, runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
