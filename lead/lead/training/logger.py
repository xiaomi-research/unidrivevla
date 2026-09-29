import logging
import os

import numpy as np
import torch
import wandb
from beartype import beartype
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from lead.plant.plant_visualizer import visualize_plant_sample
from lead.tfv6.tfv6 import Prediction
from lead.training.config_training import TrainingConfig
from lead.visualization.visualizer import visualize_sample

LOG = logging.getLogger(__name__)


class Logger:
    @beartype
    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
        continue_step: int,
        dataset: Dataset,
        dataloader: DataLoader,
        total_gradient_steps: int,
    ):
        """
        Initialize the Logger for training.

        Args:
            config: The configuration dictionary.
            model: The model to log.
            optimizer: The optimizer to log.
            scaler: The gradient scaler for mixed precision training.
            continue_step: The step to continue from if training is resumed.
            dataset: The dataset used for training.
            dataloader: The dataloader used for training.
            total_gradient_steps: Total number of gradient steps for the training.
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.step = continue_step
        self.dataset = dataset
        self.total_gradient_steps = total_gradient_steps
        self.dataloader = dataloader
        self.scaler = scaler

        # Initialize TensorBoard and WandB loggers
        self.tensorboard_writer = None
        if self.config.rank == 0 and self.config.logdir is not None:
            self.tensorboard_writer = SummaryWriter(
                log_dir=config.logdir,
            )
            if self.config.log_wandb:
                wandb.init(
                    project=self.config.wandb_project_name,
                    name=config.description,
                    config=config.training_dict(),
                    id=config.wandb_id,
                    resume=config.wandb_resume,
                    dir=self.config.logdir,
                    settings=wandb.Settings(_service_wait=300),
                )
                self.step = max(self.step, wandb.run.step)
                LOG.info(
                    f"WandB logger will log scalar every {self.config.log_scalars_frequency} steps",
                )
                LOG.info(
                    f"WandB logger will log images every {self.config.log_images_frequency} steps",
                )
            else:
                LOG.info("WandB is disabled. Only TensorBoard logging is enabled.")

    def __del__(self):
        if self.config.rank == 0:
            if self.config.log_wandb:
                wandb.finish()
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()

    @beartype
    def log_train(
        self,
        epoch_iteration: int,
        cur_epoch: int,
        unscaled_loss: dict,
        scaled_loss: dict,
        data: dict,
        step: int,
        gradient_steps_skipped: int,
        predictions: Prediction,
        log: dict,
    ):
        """
        Log training information.

        Args:
            epoch_iteration: Current iteration number of training in the epoch.
            cur_epoch: Current epoch number.
            unscaled_loss: Dictionary of unscaled losses for the current epoch.
            scaled_loss: Dictionary of scaled losses for the current epoch.
            data: Dictionary containing data used for training.
            step: Current step in the training process.
            gradient_steps_skipped: Number of gradient steps skipped due to inf/nan gradients.
            predictions: Model predictions for the current batch.
            log: Dictionary containing debug information.
        """
        is_plant = self.config.model_type == "plant"
        should_viz = (
            self.config.rank == 0
            and self.config.visualize_training
            and (
                (epoch_iteration + 1) % self.config.log_images_frequency == 0
                or epoch_iteration <= 1
            )
        )
        viz_fn = visualize_plant_sample if is_plant else visualize_sample

        if should_viz and self.config.debug_mode and self.config.carla_leaderboard_mode:
            LOG.info(f"Visualizing training sample at step {step}.")
            viz_fn(
                config=self.config,
                predictions=predictions,
                data=data,
                save_image=True,
                save_path=os.path.join("outputs", "training_viz"),
                postfix=str(self.step).zfill(5),
                prefix="train",
            )

        if self.config.rank == 0:
            if (
                self.config.log_wandb
                and should_viz
                and self.config.carla_leaderboard_mode
            ):
                LOG.info(f"Logging training sample to WandB at step {step}.")
                viz_fn(
                    config=self.config,
                    predictions=predictions,
                    data=data,
                    prefix="train",
                    log_wandb=True,
                )

            if (epoch_iteration + 1) % self.config.log_scalars_frequency == 0:
                self.step = max(self.step, step)

                message = {}

                # General logs
                message["debug/epoch"] = cur_epoch
                for g in range(len(self.optimizer.param_groups)):
                    message[f"debug/lr_{g}"] = self.optimizer.param_groups[g]["lr"]
                message["debug/batch_size_per_gpu"] = data["source_dataset"].shape[0]
                message["debug/num_gpu"] = torch.cuda.device_count()
                message["debug/model_size"] = sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                )
                message["debug/dataset_size"] = len(self.dataset)
                message["debug/num_gradient_steps"] = self.total_gradient_steps
                message["debug/finished_percentage"] = step / self.total_gradient_steps
                message["debug/steps_left"] = self.total_gradient_steps - step
                message["debug/dataloader_size"] = len(self.dataloader)
                message["debug/allocated_cpus"] = self.config.assigned_cpu_cores
                message["debug/gradient_steps_skipped"] = gradient_steps_skipped
                message["debug/max_gpu_mem"] = torch.cuda.max_memory_allocated(
                    self.config.device,
                ) / (1024**3)  # Convert to GB
                message["debug/average_loading_time"] = (
                    data["loading_time"].cpu().numpy().mean()
                )
                message["debug/average_loading_meta_time"] = (
                    data["loading_meta_time"].cpu().numpy().mean()
                )
                message["debug/average_loading_sensor_time"] = (
                    data["loading_sensor_time"].cpu().numpy().mean()
                )
                message["debug/source_dataset"] = (
                    data["source_dataset"].cpu().numpy().mean()
                )
                if self.scaler is not None:
                    message["debug/grad_scale"] = self.scaler.get_scale()

                # Loss and metrics logs
                message.update(log)
                for loss_name, loss_value in unscaled_loss.items():
                    message[f"unscaled_loss/{loss_name}"] = loss_value.float().item()
                for loss_name, loss_value in scaled_loss.items():
                    message[f"scaled_loss/{loss_name}"] = loss_value.float().item()
                for msg_name, msg_value in log.items():
                    if msg_name.startswith("metric/"):
                        mean_val = msg_value
                        if isinstance(msg_value, torch.Tensor):
                            mean_val = msg_value.detach().float().cpu().numpy().mean()
                        if isinstance(msg_value, np.ndarray):
                            mean_val = mean_val.mean()
                        message[msg_name] = mean_val

                # Convert bfloat16 to float for logging
                for key, value in message.items():
                    if (
                        isinstance(value, torch.Tensor)
                        and value.dtype == torch.bfloat16
                    ):
                        message[key] = value.float().item()

                if self.config.log_wandb:
                    wandb.log(message, step=self.step, commit=False)

                if self.tensorboard_writer is not None:
                    for key, value in message.items():
                        self.tensorboard_writer.add_scalar(
                            key,
                            value,
                            self.step,
                        )
            self.step += 1

    def logs(self, msg: dict):
        if self.config.rank == 0:
            if self.config.log_wandb:
                wandb.log(
                    msg,
                    step=self.step,
                    commit=False,
                )

            if self.tensorboard_writer is not None:
                for key, value in msg.items():
                    self.tensorboard_writer.add_scalar(
                        key,
                        value,
                        self.step,
                    )
