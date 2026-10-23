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

import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)

    def _load_lora_adapter(self, path: str):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_path = os.path.join(path, "lora_adapter", "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            print(f"[rank-{self.rank}]: No LoRA adapter found at {adapter_path}, skipping.")
            return

        print(f"[rank-{self.rank}]: Loading LoRA adapter from {os.path.abspath(adapter_path)}.")
        lora_state_dict = load_file(adapter_path, device="cpu")
        set_peft_model_state_dict(self.model._fsdp_wrapped_module, lora_state_dict)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        if self._is_peft_model:
            self._load_lora_adapter(path)
            optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
            extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
            if os.path.exists(optim_path) and os.path.exists(extra_path):
                optim_state_dict = torch.load(optim_path, weights_only=False)
                extra_state_dict = torch.load(extra_path, weights_only=False)
                state_dict_options = StateDictOptions(cpu_offload=True)
                set_state_dict(
                    model=self.model,
                    optimizers=self.optimizer,
                    model_state_dict={},
                    optim_state_dict=optim_state_dict,
                    options=state_dict_options,
                )
                self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])
                if "rng" in extra_state_dict:
                    self.load_rng_state(extra_state_dict["rng"])
            return

        # every rank download its own checkpoint
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)

        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
        print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
        optim_state_dict = torch.load(optim_path, weights_only=False)
        extra_state_dict = torch.load(extra_path, weights_only=False)

        state_dict_options = StateDictOptions(cpu_offload=True)
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=model_state_dict,
            optim_state_dict=optim_state_dict,
            options=state_dict_options,
        )
        self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])

        # recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

    @property
    def _is_peft_model(self) -> bool:
        try:
            from peft import PeftModel

            return isinstance(self.model._fsdp_wrapped_module, PeftModel)
        except ImportError:
            return False

    def _save_lora_adapter(self, path: str):
        from peft import get_peft_model_state_dict
        from torch.distributed._tensor import DTensor

        lora_state_dict = get_peft_model_state_dict(self.model._fsdp_wrapped_module)
        cpu_state_dict = {}
        for key, value in lora_state_dict.items():
            if isinstance(value, DTensor):
                cpu_state_dict[key] = value.full_tensor().detach().cpu()
            elif isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.detach().cpu()
            else:
                cpu_state_dict[key] = value

        if self.rank == 0:
            adapter_path = os.path.join(path, "lora_adapter")
            os.makedirs(adapter_path, exist_ok=True)

            from safetensors.torch import save_file

            save_file(cpu_state_dict, os.path.join(adapter_path, "adapter_model.safetensors"))

            import json

            peft_config = self.model._fsdp_wrapped_module.peft_config["default"]
            config_dict = peft_config.__dict__.copy()
            config_dict["auto_mapping"] = None
            with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

            print(f"[rank-{self.rank}]: Saved LoRA adapter to {os.path.abspath(adapter_path)}.")

    def save_checkpoint(self, path: str, save_model_only: bool = False):
        path = self.local_mkdir(path)
        dist.barrier()

        if self._is_peft_model:
            self._save_lora_adapter(path)
            if not save_model_only:
                optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
                state_dict_options = StateDictOptions(cpu_offload=True)
                _, optim_state_dict = get_state_dict(self.model, self.optimizer, options=state_dict_options)
                extra_state_dict = {
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "rng": self.get_rng_state(),
                }
                torch.save(optim_state_dict, optim_path)
                torch.save(extra_state_dict, extra_path)
            dist.barrier()
            return

        # every rank will save its own model and optim shard
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        state_dict_options = StateDictOptions(cpu_offload=True)
        if save_model_only:
            model_state_dict = get_model_state_dict(self.model, options=state_dict_options)
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            torch.save(model_state_dict, model_path)
        else:
            model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer, options=state_dict_options)
            extra_state_dict = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "rng": self.get_rng_state(),
            }
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}.")
            print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
            torch.save(model_state_dict, model_path)
            torch.save(optim_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        dist.barrier()

        if self.rank == 0:
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)
            wrapped_model = self.model._fsdp_wrapped_module
            if hasattr(wrapped_model, "get_base_model"):
                wrapped_model = wrapped_model.get_base_model()
            assert isinstance(wrapped_model, PreTrainedModel)
            wrapped_model.config.save_pretrained(hf_path)
            wrapped_model.generation_config.save_pretrained(hf_path)
            self.processing_class.save_pretrained(hf_path)

        dist.barrier()
