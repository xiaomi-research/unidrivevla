#!/usr/bin/env python
import json
import traceback
from collections import defaultdict

import wandb

from lead.inference.config_closed_loop import ClosedLoopConfig
from slurm.config_slurm import ConfigSlurm
from slurm.evaluation.evaluate_utils import did_route_crash, is_on_slurm, load_json_file


class WandBLogger:
    def __init__(
        self,
        closed_loop_config: ClosedLoopConfig,
        num_routes,
        project_name="lead_eval",
    ):
        """
        Initialize the WandBLogger, set up a run with the project name and optional job name.
        """
        self.num_routes = num_routes  # Total routes to log
        self.metrics = defaultdict(
            lambda: [0] * num_routes,
        )  # Initialize default values for metrics
        self.config_slurm = ConfigSlurm()
        if is_on_slurm():
            self.run = wandb.init(
                project=project_name,
                id=self.config_slurm.experiment_run_id,
                name=f"{self.config_slurm.experiment_run_id}",
                resume="never",
                dir=self.config_slurm.evaluation_output_dir,
                notes=self.config_slurm.evaluation_output_dir,
            )
            self.finished = False
        self.logged_ds = []

    def __del__(self):
        if is_on_slurm() and not self.finished:
            self.run.alert(
                title="Evaluation crashed",
                text="Experiment crashed before finishing.",
            )
            self.run.finish(exit_code=1)

    def log_job(
        self,
        json_log_file,
        route_idx,
        num_parallel_jobs,
        num_attempts,
        job_failed,
        carla_route_id,
    ) -> bool:
        """
        Log everything related to a job.
        """
        if is_on_slurm():
            if not job_failed or (job_failed and json_log_file.exists()):
                data = load_json_file(json_log_file)
                if data is None:
                    self.metrics["meta/crashed"][route_idx] = 1
                else:
                    try:
                        self._update_metrics(
                            data["_checkpoint"]["records"][0],
                            route_idx,
                        )
                        self.metrics["meta/crashed"][route_idx] = int(
                            did_route_crash(data),
                        )
                    except Exception as e:
                        print(
                            f"Error updating metrics for record {carla_route_id}: {e}",
                        )
                        print(traceback.format_exc())
                        self.metrics["meta/crashed"][route_idx] = 1
            elif job_failed:
                self.metrics["meta/crashed"][route_idx] = 1
            # Log the metrics for the current record index. Mostly 0.
            self.metrics["meta/num_attempts"][route_idx] = num_attempts
            self.metrics["meta/num_parallel_jobs"][route_idx] = num_parallel_jobs
            if carla_route_id.count("_") > 1:
                print(f"Route ID contains more than one underscore: {carla_route_id}")
            if self.config_slurm.evaluation_dataset in ["bench2drive", "longest6"]:
                self.run.log(
                    {"ids/route_id": float(carla_route_id.replace("_", "."))},
                    commit=False,
                )
            elif self.config_slurm.evaluation_dataset in ["Town13"]:
                self.run.log({"ids/route_id": int(carla_route_id)}, commit=False)
            elif self.config_slurm.evaluation_dataset in ["fail2drive"]:
                self.run.log(
                    {"ids/route_id": int(carla_route_id.rsplit("_", 1)[-1])},
                    commit=False,
                )
            else:
                raise ValueError(
                    f"Unknown EVALUATION_DATASET: {self.config_slurm.evaluation_dataset}",
                )
            for i, (metric_name, values) in enumerate(iterable=self.metrics.items()):
                self.run.log(
                    {metric_name: values[route_idx]},
                    commit=i == len(self.metrics) - 1,
                )
            print(f"Logged evaluation results for record {carla_route_id} to wandb")
            return bool(self.metrics["scores/success"][route_idx])

    def log_metrics_artifact(self):
        if is_on_slurm():
            artifact = wandb.Artifact(name="metrics", type="metrics")
            artifact.add_file(self.config_slurm.merged_results_path)
            self.run.log_artifact(artifact)
            print("Logged metrics artifact to wandb")
            try:
                with open(self.config_slurm.merged_results_path) as f:
                    metrics = json.load(f)
                    self.run.log(
                        {
                            f"{self.config_slurm.evaluation_dataset}/driving_score": metrics[
                                "driving score"
                            ],
                        },
                    )
                    self.run.log(
                        {
                            f"{self.config_slurm.evaluation_dataset}/success_rate": metrics[
                                "success rate"
                            ],
                        },
                    )
                    self.run.log(
                        {
                            f"{self.config_slurm.evaluation_dataset}/eval_num": metrics[
                                "eval num"
                            ],
                        },
                    )
            except Exception as e:
                print(f"Error loading metrics from merged.json: {e}")

    def finish(self, exit_code=0):
        if is_on_slurm():
            self.run.alert(title="Evaluation finished", text="Experiment finished.")
            self.finished = True
            self.run.finish(exit_code=exit_code)

    def _update_metrics(self, record, route_idx):
        """
        Update the metrics array with details from a single record for a specific route index.
        """
        infractions = record.get("infractions", {})
        scores = record.get("scores", {})
        meta = record.get("meta", {})

        for key, value in infractions.items():
            self.metrics[f"infractions/{key}"][route_idx] = len(value)

        for key, value in scores.items():
            self.metrics[f"scores/{key}"][route_idx] = value
        self.metrics["scores/success"][route_idx] = int(
            (scores.get("score_penalty", 0.0) == 1.0)
            and (scores.get("score_composed", 0.0) == 100.0),
        )  # success = perfect score

        for key, value in meta.items():
            self.metrics[f"meta/{key}"][route_idx] = value

        self.metrics["ids/town"][route_idx] = int(
            record.get("town_name", "Town999").replace("Town", "").replace("HD", ""),
        )
        self.logged_ds.append(self.metrics["scores/score_composed"][route_idx])
        self.metrics["scores/running_avg"][route_idx] = sum(self.logged_ds) / len(
            self.logged_ds,
        )
