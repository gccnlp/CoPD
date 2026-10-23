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
A unified tracking interface that supports logging data to different backend
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch

from ..py_functional import convert_dict_to_str, flatten_dict, is_package_available, unflatten_dict
from .gen_logger import AggregateGenerationsLogger, GenerationSample


if is_package_available("mlflow"):
    import mlflow  # type: ignore


if is_package_available("tensorboard"):
    from torch.utils.tensorboard import SummaryWriter


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


class Logger(ABC):
    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None: ...

    @abstractmethod
    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None: ...

    def finish(self) -> None:
        pass


class ConsoleLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        print("Config\n" + convert_dict_to_str(config))

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        prefix = f"Step {step}" if step is not None else "Log"
        print(f"{prefix}\n" + convert_dict_to_str(unflatten_dict(data)))


class FileLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        print(f"Initializing logging file to {config['trainer']['save_checkpoint_path']}.")
        os.makedirs(config["trainer"]["save_checkpoint_path"], exist_ok=True)
        with open(os.path.join(config["trainer"]["save_checkpoint_path"], "experiment_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(config["trainer"]["save_checkpoint_path"], "experiment_log.jsonl"), "w") as f:
            pass

        with open(os.path.join(config["trainer"]["save_checkpoint_path"], "generations.log"), "w") as f:
            pass

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        record = unflatten_dict(data)
        if step is not None:
            record = {"step": step, **record}
        with open(os.path.join(self.config["trainer"]["save_checkpoint_path"], "experiment_log.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")


class MlflowLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        mlflow.start_run(run_name=config["trainer"]["experiment_name"])
        mlflow.log_params(flatten_dict(config))

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics=data, step=step)


class SwanlabLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        swanlab_key = os.getenv("SWANLAB_API_KEY")
        swanlab_dir = os.getenv("SWANLAB_DIR", "swanlab_log")
        swanlab_mode = os.getenv("SWANLAB_MODE", "cloud")
        if swanlab_key:
            swanlab.login(swanlab_key)

        swanlab.init(
            project=config["trainer"]["project_name"],
            experiment_name=config["trainer"]["experiment_name"],
            config={"UPPERFRAMEWORK": "EasyR1", "FRAMEWORK": "veRL", **config},
            logdir=swanlab_dir,
            mode=swanlab_mode,
        )

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        swanlab.log(data=data, step=step)

    def finish(self) -> None:
        swanlab.finish()


class TensorBoardLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        tensorboard_dir = os.getenv("TENSORBOARD_DIR", "tensorboard_log")
        tensorboard_dir = os.path.join(
            tensorboard_dir, config["trainer"]["project_name"], config["trainer"]["experiment_name"]
        )
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)
        config_dict = {}
        for key, value in flatten_dict(config).items():
            if isinstance(value, (int, float, str, bool, torch.Tensor)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)

        self.writer.add_hparams(hparam_dict=config_dict, metric_dict={"placeholder": 0})

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


class WandbLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        self.branch_metric_names = _get_coev_branch_metric_names(config)
        wandb.init(
            project=config["trainer"]["project_name"],
            name=config["trainer"]["experiment_name"],
            config=config,
        )
        wandb.define_metric(f"{self.branch_metric_names['branch_text']}_step")
        wandb.define_metric(f"{self.branch_metric_names['branch_vision']}_step")
        wandb.define_metric("train_global_step")
        # Bind branch metrics to their own step counters.
        # IMPORTANT: define the step fields AFTER the wildcards so that
        # branch_text_step and branch_vision_step are NOT self-referencing
        # (the last define_metric call for a given metric name wins).
        wandb.define_metric(
            f"{self.branch_metric_names['branch_text']}/*",
            step_metric=f"{self.branch_metric_names['branch_text']}_step",
        )
        wandb.define_metric(
            f"{self.branch_metric_names['branch_vision']}/*",
            step_metric=f"{self.branch_metric_names['branch_vision']}_step",
        )
        wandb.define_metric("coev/*", step_metric="train_global_step")

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        data = self._remap_branch_metrics(data)
        if step is not None:
            wandb.log(data=data, step=step)
        else:
            wandb.log(data=data)

    def finish(self) -> None:
        wandb.finish()

    def _remap_branch_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        remapped = {}
        for key, value in data.items():
            remapped[self._remap_branch_metric_key(key)] = value
        return remapped

    def _remap_branch_metric_key(self, key: str) -> str:
        for internal_name, display_name in self.branch_metric_names.items():
            step_key = f"{internal_name}_step"
            if key == step_key:
                return f"{display_name}_step"
            if key.startswith(f"{internal_name}/"):
                return f"{display_name}/{key[len(internal_name) + 1:]}"
        return key


def _get_coev_branch_metric_names(config: dict[str, Any]) -> dict[str, str]:
    coev_config = config.get("coev", {})
    branch_metric_names = {
        "branch_text": ("wandb_branch_name_text", coev_config.get("wandb_branch_name_text", "branch_text")),
        "branch_vision": ("wandb_branch_name_vision", coev_config.get("wandb_branch_name_vision", "branch_vision")),
    }
    resolved_metric_names = {}
    for internal_name, (config_key, value) in branch_metric_names.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`coev.{config_key}` must be a non-empty string.")
        if "/" in value:
            raise ValueError(f"Wandb branch metric name `{value}` must not contain `/`.")
        resolved_metric_names[internal_name] = value.strip()
    return resolved_metric_names


LOGGERS = {
    "console": ConsoleLogger,
    "file": FileLogger,
    "mlflow": MlflowLogger,
    "swanlab": SwanlabLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandbLogger,
}


class Tracker:
    def __init__(self, loggers: Union[str, list[str]] = "console", config: Optional[dict[str, Any]] = None):
        if isinstance(loggers, str):
            loggers = [loggers]

        self.loggers: list[Logger] = []
        for logger in loggers:
            if logger not in LOGGERS:
                raise ValueError(f"{logger} is not supported.")

            self.loggers.append(LOGGERS[logger](config))

        self.gen_logger = AggregateGenerationsLogger(loggers, config)

    def log(self, data: dict[str, Any], step: Optional[int] = None) -> None:
        for logger in self.loggers:
            logger.log(data=data, step=step)

    def log_generation(self, samples: list[GenerationSample], step: int) -> None:
        self.gen_logger.log(samples, step)

    def __del__(self):
        for logger in self.loggers:
            logger.finish()
