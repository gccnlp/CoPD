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

from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


def _create_dataset(
    data_path: str,
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
) -> RLHFDataset:
    return RLHFDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        video_max_frames=config.video_max_frames,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
        video_min_pixels=config.video_min_pixels,
        video_max_pixels=config.video_max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        use_preprocessed_videos=config.use_preprocessed_videos,
        preprocessed_video_dir=config.preprocessed_video_dir,
    )


class SequentialThenRandomSampler(Sampler[int]):
    """First epoch yields indices sequentially; subsequent epochs yield random permutations."""

    def __init__(self, data_source: Sized, generator: torch.Generator = None):
        self.data_source = data_source
        self.generator = generator
        self._epoch = 0

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self._epoch == 0:
            yield from range(n)
        else:
            if self.generator is None:
                yield from torch.randperm(n).tolist()
            else:
                yield from torch.randperm(n, generator=self.generator).tolist()
        self._epoch += 1

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self):
        return {"_epoch": self._epoch}

    def load_state_dict(self, state_dict):
        self._epoch = state_dict.get("_epoch", 0)


def _create_train_val_dataloaders(
    train_files: str,
    val_files: str,
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    name: str = "",
    train_batch_size_override: Optional[int] = None,
    shuffle_override: Optional[bool] = None,
    sequential_first: bool = False,
):
    train_dataset = _create_dataset(train_files, config, tokenizer, processor)
    # use sampler for better ckpt resume
    shuffle = shuffle_override if shuffle_override is not None else config.shuffle
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(config.seed)
    if sequential_first:
        sampler = SequentialThenRandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    elif shuffle:
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if train_batch_size_override is not None:
        train_batch_size = train_batch_size_override
    elif config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    if not val_files:
        if len(train_dataloader) < 1:
            raise ValueError("Training data must contain at least one complete rollout batch.")
        return train_dataloader, None

    val_dataset = _create_dataset(val_files, config, tokenizer, processor)

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    prefix = f"{name} " if name else ""
    print(f"Size of {prefix}train dataloader: {len(train_dataloader)}")
    print(f"Size of {prefix}val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader


def create_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    name: str = "",
    train_batch_size_override: Optional[int] = None,
    shuffle_override: Optional[bool] = None,
    sequential_first: bool = False,
) -> None:
    return _create_train_val_dataloaders(
        train_files=config.train_files,
        val_files=config.val_files,
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        name=name,
        train_batch_size_override=train_batch_size_override,
        shuffle_override=shuffle_override,
        sequential_first=sequential_first,
    )


def create_coevol_dataloaders(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    rollout_batch_size_text: Optional[int] = None,
    rollout_batch_size_vision: Optional[int] = None,
    shuffle_text: Optional[bool] = None,
    shuffle_vision: Optional[bool] = None,
    sequential_first_text: bool = False,
    sequential_first_vision: bool = False,
):
    required_fields = {
        "text_train_files": config.text_train_files,
        "vision_train_files": config.vision_train_files,
    }
    missing = [name for name, value in required_fields.items() if not value]
    if missing:
        raise ValueError(f"Co-evolution training requires data fields: {', '.join(missing)}.")

    shared_train_batch_size = max(
        rollout_batch_size_text if rollout_batch_size_text is not None else config.rollout_batch_size,
        rollout_batch_size_vision if rollout_batch_size_vision is not None else config.rollout_batch_size,
    )

    text_train_dataloader, text_val_dataloader = _create_train_val_dataloaders(
        train_files=config.text_train_files,
        val_files=config.text_val_files,
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        name="text",
        train_batch_size_override=shared_train_batch_size,
        shuffle_override=shuffle_text,
        sequential_first=sequential_first_text,
    )
    vision_train_dataloader, vision_val_dataloader = _create_train_val_dataloaders(
        train_files=config.vision_train_files,
        val_files=config.vision_val_files,
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        name="vision",
        train_batch_size_override=shared_train_batch_size,
        shuffle_override=shuffle_vision,
        sequential_first=sequential_first_vision,
    )
    return text_train_dataloader, text_val_dataloader, vision_train_dataloader, vision_val_dataloader
