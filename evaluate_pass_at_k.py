#!/usr/bin/env python3
"""
Script to run SWE-bench evaluation for multiple prediction runs and calculate pass@k metrics.

This script:
1. Runs evaluation for each prediction file in runs/run_1 through runs/run_5
2. Collects results from the evaluation logs
3. Calculates pass@k (where k=5) for each dataset entry

Note: Docker images from revelotalentcorp DockerHub will be automatically pulled if not found locally.
Ensure Docker is running and authenticated if the images are private.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import docker

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, LOG_REPORT, RUN_EVALUATION_LOG_DIR


def get_model_name_from_predictions_file(predictions_path: Path) -> str:
    """Extract model name from predictions file name."""
    # File format: predictions_<model_name>.jsonl
    filename = predictions_path.name
    if filename.startswith("predictions_") and filename.endswith(".jsonl"):
        model_name = filename[len("predictions_"):-len(".jsonl")]
        return model_name
    raise ValueError(f"Unexpected predictions file name format: {filename}")


def run_evaluation(
    dataset_path: str,
    predictions_path: str,
    run_id: str,
    split: str = "test",
) -> bool:
    """
    Run SWE-bench evaluation for a single predictions file.
    
    Note: Docker images specified in the dataset's install_config.docker_image
    (e.g., revelotalentcorp/* images) will be automatically pulled from DockerHub
    if not found locally. The evaluation harness handles this automatically.
    
    Returns True if successful, False otherwise.
    """
    cmd = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_path,
        "--split",
        split,
        "--predictions_path",
        predictions_path,
        "--run_id",
        run_id,
    ]
    
    print(f"\n{'='*80}")
    print(f"Running evaluation for {run_id}")
    print(f"Predictions file: {predictions_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
        )
        print(f"\n✓ Evaluation completed successfully for {run_id}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed for {run_id}: {e}\n")
        return False


def load_dataset_instances(dataset_path: str) -> List[str]:
    """Load instance IDs from the dataset file."""
    instance_ids = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                instance = json.loads(line)
                instance_ids.append(instance[KEY_INSTANCE_ID])
    return instance_ids


def collect_evaluation_results(
    run_ids: List[str],
    model_name: str,
    instance_ids: List[str],
) -> Dict[str, List[bool]]:
    """
    Collect resolved status for each instance from all evaluation runs.
    
    Returns a dictionary mapping instance_id to a list of resolved statuses
    (one per run, in order).
    """
    results: Dict[str, List[bool]] = {instance_id: [] for instance_id in instance_ids}
    
    for run_id in run_ids:
        for instance_id in instance_ids:
            # Report path: logs/run_evaluation/{run_id}/{model_name}/{instance_id}/report.json
            report_path = (
                RUN_EVALUATION_LOG_DIR
                / run_id
                / model_name.replace("/", "__")
                / instance_id
                / LOG_REPORT
            )
            
            resolved = False
            if report_path.exists():
                try:
                    with open(report_path, "r") as f:
                        report = json.load(f)
                        if instance_id in report:
                            resolved = report[instance_id].get("resolved", False)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse report for {instance_id} in {run_id}: {e}")
            
            results[instance_id].append(resolved)
    
    return results


def calculate_pass_at_k(
    results: Dict[str, List[bool]],
    k: int = 5,
) -> Dict[str, int]:
    """
    Calculate pass@k for each instance.
    
    Returns a dictionary mapping instance_id to the number of successful resolutions
    out of k attempts.
    """
    pass_at_k = {}
    for instance_id, resolved_list in results.items():
        if len(resolved_list) != k:
            print(f"Warning: Instance {instance_id} has {len(resolved_list)} results, expected {k}")
        # Count how many times it was resolved
        pass_count = sum(1 for resolved in resolved_list if resolved)
        pass_at_k[instance_id] = pass_count
    return pass_at_k


def check_docker_setup():
    """Check if Docker is running and accessible."""
    try:
        client = docker.from_env()
        client.ping()
        print("✓ Docker is running and accessible")
        return True
    except docker.errors.DockerException as e:
        print(f"✗ Docker is not accessible: {e}")
        print("  Please ensure Docker is running and accessible.")
        return False
    except Exception as e:
        print(f"✗ Error checking Docker: {e}")
        return False


def main():
    """Main execution function."""
    # Configuration
    dataset_path = "dataset_batch2.jsonl"
    runs_dir = Path("runs")
    num_runs = 5
    k = 5
    
    # Check Docker setup
    print("Checking Docker setup...")
    if not check_docker_setup():
        print("\n⚠️  Warning: Docker may not be properly configured.")
        print("  Docker images from revelotalentcorp DockerHub will be pulled automatically,")
        print("  but this requires Docker to be running and authenticated if images are private.")
        print("  Continuing anyway...\n")
    
    # Validate inputs
    if not Path(dataset_path).exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        sys.exit(1)
    
    # Find all prediction files and extract model name
    predictions_files = []
    model_name = None
    
    for i in range(1, num_runs + 1):
        run_dir = runs_dir / f"run_{i}"
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            sys.exit(1)
        
        # Find predictions file
        pred_files = list(run_dir.glob("predictions_*.jsonl"))
        if not pred_files:
            print(f"Error: No predictions file found in {run_dir}")
            sys.exit(1)
        if len(pred_files) > 1:
            print(f"Warning: Multiple predictions files found in {run_dir}, using first one")
        
        pred_file = pred_files[0]
        predictions_files.append((i, pred_file))
        
        # Extract and verify model name consistency
        current_model_name = get_model_name_from_predictions_file(pred_file)
        if model_name is None:
            model_name = current_model_name
        elif model_name != current_model_name:
            print(f"Warning: Model name mismatch. Expected {model_name}, found {current_model_name}")
    
    print(f"Found model: {model_name}")
    print(f"Found {len(predictions_files)} prediction files:")
    for run_num, pred_file in predictions_files:
        print(f"  - Run {run_num}: {pred_file}")
    print()
    
    # Load dataset to get instance IDs
    print("Loading dataset...")
    instance_ids = load_dataset_instances(dataset_path)
    print(f"Found {len(instance_ids)} instances in dataset\n")
    
    # Run evaluations
    run_ids = []
    for run_num, predictions_path in predictions_files:
        run_id = f"evaluation_{run_num}"
        run_ids.append(run_id)
        
        success = run_evaluation(
            dataset_path=dataset_path,
            predictions_path=str(predictions_path),
            run_id=run_id,
        )
        
        if not success:
            print(f"Warning: Evaluation for {run_id} may have failed. Continuing anyway...")
    
    print(f"\n{'='*80}")
    print("All evaluations completed. Collecting results...")
    print(f"{'='*80}\n")
    
    # Collect results
    results = collect_evaluation_results(run_ids, model_name, instance_ids)
    
    # Calculate pass@k
    pass_at_k = calculate_pass_at_k(results, k=k)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Pass@{k} Results")
    print(f"{'='*80}\n")
    
    # Sort by instance_id for consistent output
    sorted_instances = sorted(pass_at_k.items())
    
    print(f"{'Instance ID':<60} {'Pass@{k}':<10}")
    print("-" * 80)
    
    for instance_id, pass_count in sorted_instances:
        print(f"{instance_id:<60} {pass_count}/{k}")
    
    # Summary statistics
    total_instances = len(pass_at_k)
    instances_with_at_least_one_pass = sum(1 for count in pass_at_k.values() if count > 0)
    instances_with_all_passes = sum(1 for count in pass_at_k.values() if count == k)
    
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}")
    print(f"Total instances: {total_instances}")
    print(f"Instances with at least 1 pass: {instances_with_at_least_one_pass} ({100*instances_with_at_least_one_pass/total_instances:.1f}%)")
    print(f"Instances with all {k} passes: {instances_with_all_passes} ({100*instances_with_all_passes/total_instances:.1f}%)")
    
    # Calculate average pass@k
    avg_pass_at_k = sum(pass_at_k.values()) / total_instances if total_instances > 0 else 0
    print(f"Average pass@{k}: {avg_pass_at_k:.2f}/{k}")
    print(f"{'='*80}\n")
    
    # Save results to JSON file
    output_file = "pass_at_k_results.json"
    output_data = {
        "k": k,
        "model_name": model_name,
        "results": pass_at_k,
        "summary": {
            "total_instances": total_instances,
            "instances_with_at_least_one_pass": instances_with_at_least_one_pass,
            "instances_with_all_passes": instances_with_all_passes,
            "average_pass_at_k": avg_pass_at_k,
        },
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

