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
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    text_train_files: str = ""
    text_val_files: str = ""
    vision_train_files: str = ""
    vision_val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    video_max_frames: int = 128
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    image_min_pixels: Optional[int] = None
    image_max_pixels: Optional[int] = None
    video_min_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    use_preprocessed_videos: bool = True
    """whether to use preprocessed video files (.pt) if available"""
    preprocessed_video_dir: Optional[str] = None
    """directory containing preprocessed video files (.pt)"""

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")
        self.preprocessed_video_dir = get_abs_path(self.preprocessed_video_dir, prompt="Preprocessed video directory")
        if self.image_min_pixels is None:
            self.image_min_pixels = self.min_pixels
        if self.image_max_pixels is None:
            self.image_max_pixels = self.max_pixels
        if self.video_min_pixels is None:
            self.video_min_pixels = self.min_pixels
        if self.video_max_pixels is None:
            self.video_max_pixels = self.max_pixels


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_metric_grouping: str = "auto"
    """validation metric grouping: `auto`, `data_type`, `problem_type`, or `both`"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")
        if self.val_metric_grouping not in {"auto", "data_type", "problem_type", "both"}:
            raise ValueError(
                "`trainer.val_metric_grouping` must be one of: "
                "`auto`, `data_type`, `problem_type`, `both`."
            )


@dataclass
class ParallelConfig:
    enabled: bool = False
    """enable dual-model parallel training"""
    n_gpus_per_model: Optional[int] = None
    """Global fallback GPUs per branch. Dual-branch trainers still use this field directly."""
    n_gpus_per_model_hub: Optional[int] = None
    """Tri-branch hub topology: GPUs for the configured hub branch. None = default to 4 when hub mode is enabled."""
    n_gpus_per_model_leaf: Optional[int] = None
    """Tri-branch hub topology: GPUs for non-hub branches. None = default to 2 when hub mode is enabled."""
    n_gpus_per_model_text: Optional[int] = None
    """Tri-branch trainer: GPUs for the text branch. None = fall back to n_gpus_per_model."""
    n_gpus_per_model_vision: Optional[int] = None
    """Tri-branch trainer: GPUs for the vision branch. None = fall back to n_gpus_per_model."""
    n_gpus_per_model_video: Optional[int] = None
    """Tri-branch trainer: GPUs for the video branch. None = fall back to n_gpus_per_model."""
    rollout_seed_a: Optional[int] = None
    rollout_seed_b: Optional[int] = None
    rollout_seed_c: Optional[int] = None
    rollout_temperature_a: Optional[float] = None
    rollout_temperature_b: Optional[float] = None
    rollout_temperature_c: Optional[float] = None
    rollout_top_p_a: Optional[float] = None
    rollout_top_p_b: Optional[float] = None
    rollout_top_p_c: Optional[float] = None
    rollout_batch_size_a: Optional[int] = None
    """per-branch rollout batch size for branch A (text). Defaults to data.rollout_batch_size."""
    rollout_batch_size_b: Optional[int] = None
    """per-branch rollout batch size for branch B (vision). Defaults to data.rollout_batch_size."""
    rollout_batch_size_c: Optional[int] = None
    """per-branch rollout batch size for branch C (video). Defaults to data.rollout_batch_size."""
    actor_global_batch_size_a: Optional[int] = None
    """per-branch actor global batch size for branch A (text). Defaults to worker.actor.global_batch_size."""
    actor_global_batch_size_b: Optional[int] = None
    """per-branch actor global batch size for branch B (vision). Defaults to worker.actor.global_batch_size."""
    actor_global_batch_size_c: Optional[int] = None
    """per-branch actor global batch size for branch C (video). Defaults to worker.actor.global_batch_size."""
    critic_global_batch_size_a: Optional[int] = None
    """per-branch critic global batch size for branch A (text). Defaults to worker.critic.global_batch_size."""
    critic_global_batch_size_b: Optional[int] = None
    """per-branch critic global batch size for branch B (vision). Defaults to worker.critic.global_batch_size."""
    critic_global_batch_size_c: Optional[int] = None
    """per-branch critic global batch size for branch C (video). Defaults to worker.critic.global_batch_size."""
    peer_replacement_count: int = 1
    """how many peer responses to inject into each branch per prompt"""
    model_path_a: Optional[str] = None
    """HuggingFace model for branch A (text). None = use worker.actor.model.model_path."""
    model_path_b: Optional[str] = None
    """HuggingFace model for branch B (vision). None = use worker.actor.model.model_path."""
    model_path_c: Optional[str] = None
    """HuggingFace model for branch C (video). None = use worker.actor.model.model_path."""
    shuffle_a: Optional[bool] = None
    """Shuffle training data for branch A (text). None = fall back to data.shuffle."""
    shuffle_b: Optional[bool] = None
    """Shuffle training data for branch B (vision). None = fall back to data.shuffle."""
    shuffle_c: Optional[bool] = None
    """Shuffle training data for branch C (video). None = fall back to data.shuffle."""
    sequential_first_a: bool = False
    """Branch A (text): first epoch sequential, then shuffle. Takes priority over shuffle_a."""
    sequential_first_b: bool = False
    """Branch B (vision): first epoch sequential, then shuffle. Takes priority over shuffle_b."""
    sequential_first_c: bool = False
    """Branch C (video): first epoch sequential, then shuffle. Takes priority over shuffle_c."""
    nodes_text: Optional[int] = None
    """Machine-level allocation: number of nodes dedicated to the text branch. Each node uses all GPUs."""
    nodes_vision: Optional[int] = None
    """Machine-level allocation: number of nodes dedicated to the vision branch. Each node uses all GPUs."""
    nodes_video: Optional[int] = None
    """Machine-level allocation: number of nodes dedicated to the video branch. Each node uses all GPUs."""


@dataclass
class CoEvolConfig:
    enabled: bool = False
    """enable co-evolution training with separate text and vision branches"""
    wandb_branch_name_text: str = "branch_text"
    """wandb metric prefix for the internal `branch_text` branch"""
    wandb_branch_name_vision: str = "branch_vision"
    """wandb metric prefix for the internal `branch_vision` branch"""
    mode: str = "cross_reward"
    """Co-evolution mode: `cross_reward` for mutual on-policy distillation."""
    anchor_coef: float = 1e-2
    """cross_reward mode: reward bonus coefficient for Branch T log-prob scoring"""
    foreign_grpo_loss_enabled: bool = True
    """enable native task reward/GRPO signal on foreign data in cross_reward mode"""
    cross_reward_loss_enabled: bool = True
    """enable peer-scoring loss in cross_reward mode; disable for native/foreign data ablations"""
    cross_reward_style: str = "reward_bonus"
    """'reward_bonus', 'reverse_kl', or 'reverse_kl_positive' (only distill tokens where teacher > student)."""
    reverse_kl_coef: float = 0.1
    """For reverse_kl styles: mixes peer-derived token advantages into adv_orig. Controls teacher trust."""
    reverse_kl_coef_text: Optional[float] = None
    """Branch A/text student coefficient for reverse_kl styles. None = fall back to reverse_kl_coef."""
    reverse_kl_coef_vision: Optional[float] = None
    """Branch B/vision student coefficient for reverse_kl styles. None = fall back to reverse_kl_coef."""
    opd_adaptive_coef: bool = False
    """Auto-zero reverse_kl_coef when the teacher branch scores lower than the student on teacher's native data."""
    freeze_zero_opd_branches: bool = False
    """Freeze branches whose effective OPD/reverse-KL coefficient is exactly 0.0."""
    pure_opd_text: bool = False
    """Dual-branch cross_reward only: train text branch on foreign OPD data only, skipping native text data."""
    pure_opd_vision: bool = False
    """Dual-branch cross_reward only: train vision branch on foreign OPD data only, skipping native vision data."""
    opd_off_steps: int = 0
    """Number of pure-GRPO steps per OPD cycle. 0 = no cycling (use cross_reward_loss_enabled as-is)."""
    opd_on_steps: int = 0
    """Number of GRPO+OPD steps per OPD cycle. 0 = no cycling."""
    force_final_opd: bool = False
    """Force final-OPD mode for every step. Useful when hot-starting from a checkpoint."""
    hub_branch: Optional[str] = None
    """Tri-branch only: branch name or role (`text`/`vision`/`video`) to use as the hub for leaves <-> hub co-evolution. None = full graph."""


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    coev: CoEvolConfig = field(default_factory=CoEvolConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
