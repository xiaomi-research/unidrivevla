#!/usr/bin/env python
import json
import os
import shutil
import subprocess
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from merge_route_json import merge_route_json
except ModuleNotFoundError:
    from slurm.evaluation.merge_route_json import merge_route_json


SOFTWARE_FAILURE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
    "Failed - Agent's sensors were invalid",
}


def is_on_slurm():
    if os.getenv("LEAD_FORCE_LOCAL", "0") == "1":
        return False
    return shutil.which("sbatch") is not None


def did_record_crash(record):
    """Return whether a route record represents an infrastructure failure."""
    return record.get("status") in SOFTWARE_FAILURE_STATUSES


def did_route_crash(record):
    """Return whether an evaluation result needs an infrastructure retry."""
    try:
        if record.get("entry_status") != "Finished" or record.get("eligible") is not True:
            return True
        records = record["_checkpoint"]["records"]
        return len(records) == 0 or any(did_record_crash(record) for record in records)
    except Exception as e:
        print(f"Error determining if record crashed: {e}")
        print(traceback.format_exc())
        return True


def load_json_file(json_file):
    """Load a JSON file and return the content."""
    json_file = Path(json_file)
    if json_file.exists():
        json_content = None
        try:
            with open(json_file) as log_file:
                data = json.load(log_file)
                assert len(data["_checkpoint"]["records"]) > 0
                json_content = log_file.read()
            return data
        except Exception as e:
            print(f"Error reading log file {json_file}: {e}")
            print(traceback.format_exc())
            print(f"JSON Content: {json_content}")
    return None


def get_max_num_attempts():
    return int(
        open(
            f"slurm/configs/max_num_attempts_{os.getenv('EVALUATION_DATASET')}.txt",
        ).read(),
    )


def get_max_parallel_jobs():
    max_parallel_jobs = os.getenv("LEAD_LOCAL_MAX_PARALLEL_JOBS")
    if max_parallel_jobs:
        return int(max_parallel_jobs)
    if not is_on_slurm():
        return 1
    return int(
        open(
            f"slurm/configs/max_num_parallel_jobs_{os.getenv('EVALUATION_DATASET')}.txt",
        ).read(),
    )


def get_num_running_jobs(job_name, username=None):
    if username is None:
        username = os.getenv("USER")
    # On local PC
    if not is_on_slurm():
        try:
            return int(
                subprocess.check_output(
                    (f"ps -u $USER -o pid,command | grep '{job_name}' | wc -l"),
                    shell=True,
                )
                .decode("utf-8")
                .strip(),
            )
        except subprocess.CalledProcessError as e:
            print(f"Error getting number of running jobs: {e}")
            print(traceback.format_exc())
            return 0
    # On Slurm
    try:
        return int(
            subprocess.check_output(
                (
                    "SQUEUE_FORMAT2='username:15,name:100,state:10' squeue --sort V | grep"
                    f" {username} | grep {job_name} | grep 'RUNNING' | wc -l"
                ),
                shell=True,
            )
            .decode("utf-8")
            .strip(),
        )
    except subprocess.CalledProcessError as e:
        print(f"Error getting number of running jobs: {e}")
        print(traceback.format_exc())
        return 0


def submit_job(
    job_name,
    script_path,
    num_attempt,
    local_gpu_id=None,
    local_startup_delay=0,
):
    """Submit a job script using sbatch and return the job ID."""
    slurm_print_output_path = Path(script_path).parent.parent
    route_id = Path(script_path).stem

    # Define names and outputs
    job_route_name = f"{job_name}_{route_id}"
    std_err_file = (
        f"{slurm_print_output_path}/stderr/stderr_{route_id}_{num_attempt}.txt"
    )
    std_out_file = (
        f"{slurm_print_output_path}/stdout/stdout_{route_id}_{num_attempt}.txt"
    )
    os.makedirs(f"{slurm_print_output_path}/stderr", exist_ok=True)
    os.makedirs(f"{slurm_print_output_path}/stdout", exist_ok=True)
    print(std_err_file)
    print(std_out_file)

    if not is_on_slurm():
        env = os.environ.copy()
        if local_gpu_id is not None:
            env["LEAD_ASSIGNED_LOCAL_GPU"] = str(local_gpu_id)
            env["LEAD_ASSIGNED_LOCAL_STARTUP_DELAY"] = str(
                local_startup_delay)
        with open(std_out_file, "w") as stdout, open(std_err_file, "w") as stderr:
            process = subprocess.Popen(
                ["bash", str(script_path)],
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
        return process.pid, process

    extra = ""
    # If the job has failed multiple times, most properly because of memory issues, switch to a more powerful partition
    if num_attempt > 4:
        partition = "L40Sday" if "TCML" in os.environ else "a100-galvani"
        extra = f"--partition {partition} --gres gpu:1"

    submit_command = f"sbatch --job-name {job_route_name} --output {std_out_file} --error {std_err_file} {extra} {script_path}"

    # Submit jobs and return job ID
    try:
        output = subprocess.check_output(submit_command, shell=True)
        pid = output.decode("utf-8").strip().split()[-1]
        return pid, None
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print("Check script path, partition, or job name for issues.")
        raise e
    except IndexError as e:
        print("Unexpected output format from sbatch command.")
        raise e


def did_job_fail(json_log_path):
    """Determine if a job needs resubmission based on the JSON log file content."""
    json_content = load_json_file(json_log_path)
    if json_content is not None:
        return did_route_crash(json_content)
    return True


def is_job_running(slurm_job_id, slurm_job_status, job_name):
    """Check if a job is still running in the SLURM queue."""
    if not is_on_slurm():
        try:
            process = slurm_job_status["process"]
            process.wait(timeout=0)  # Wait for the process with zero timeout
            return False  # If wait completes, the process is not running
        except subprocess.TimeoutExpired:
            return True  # Timeout means the process is still running
    try:
        subprocess.check_output(f"squeue | grep {slurm_job_id}", shell=True)
        return True
    except subprocess.CalledProcessError:
        return False


def aggregate_metrics():
    """Aggregate metrics for all routes."""
    eval_output_dir = os.getenv("EVALUATION_OUTPUT_DIR")
    eval_dataset = os.getenv("EVALUATION_DATASET")
    merge_route_json(eval_output_dir + "/eval")
    shutil.copyfile(
        eval_output_dir + "/eval/merged.json",
        eval_output_dir + "/merged.json",
    )
    try:
        if "bench2drive" not in eval_dataset:
            eval_output = Path(eval_output_dir)
            route_all_folder = os.getenv("ROUTE_ALL_FOLDER")
            route_xml_candidates = [
                Path(route_all_folder) / f"{eval_dataset}.xml" if route_all_folder else None,
                eval_output / "routes" / f"{eval_dataset}.xml",
                eval_output / "routes_all" / f"{eval_dataset}.xml",
                Path("data") / "benchmark_routes" / f"{eval_dataset}.xml",
                Path("data") / f"{eval_dataset}.xml",
            ]
            route_xml = next((p for p in route_xml_candidates if p is not None and p.exists()), None)
            route_dir = Path(route_all_folder) if route_all_folder else (eval_output / "routes_all")
            if not route_dir.is_dir():
                route_dir = eval_output / "routes"
            if route_xml is None and route_dir.is_dir():
                route_files = sorted(
                    p for p in route_dir.glob("*.xml") if p.name != f"{eval_dataset}.xml"
                )
                if route_files:
                    route_xml = route_dir / f"{eval_dataset}.xml"
                    root = ET.Element("routes")
                    for route_file in route_files:
                        route_root = ET.parse(route_file).getroot()
                        for route in route_root.iter("route"):
                            root.append(route)
                    ET.ElementTree(root).write(route_xml, encoding="utf-8", xml_declaration=True)
            if route_xml is None:
                raise FileNotFoundError(
                    "Could not find route XML. Tried: "
                    + ", ".join(str(p) for p in route_xml_candidates if p is not None)
                )
            os.system(
                f"""python3 scripts/tools/result_parser.py --xml {route_xml} --results {eval_output_dir}/eval""",  # noqa: E501
            )
            shutil.copyfile(
                eval_output_dir + "/eval/results.csv",
                eval_output_dir + "/results.csv",
            )
    except Exception as e:
        print(f"Error aggregating metrics: {e}")
        print(traceback.format_exc())
        print("Could not aggregate metrics.")
