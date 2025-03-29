import argparse
import itertools
import multiprocessing
import os
import subprocess
import time

def run_benchmark(activity):
    """Run benchmark.py with the given permutation as required_activities."""
    required_activities = " ".join(map(str, activity))
    command = f"python benchmark.py --num_configs 1 --required_activities {required_activities} --num_activities {len(activity)} --benchmark_name {len(activity)}_activities{'_so' if args.use_structured_output else ''}{' --use_structured_output' if args.use_structured_output else ''}"
    print(f"Running: {command}")

    with open(os.path.join(folder, "benchmark.log"), "a") as log_file:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_file.write(f"Command: {command}\nstdout:\n{result.stdout.decode()}\nstderr:\n{result.stderr.decode()}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for the scenario generation methods.")
    parser.add_argument("--num_activities", type=int, help="The number of activities to include in the benchmark.", default=3)
    parser.add_argument("--use_structured_output", action="store_true", help="Use structured output for the LLM.", default=False)
    args = parser.parse_args()

    folder = os.path.join(os.path.dirname(__file__), "data/results")
    if not os.path.exists(folder): os.makedirs(folder)

    # Generate the first 5 permutations of numbers 0 to 5
    activities = list(itertools.permutations(range(6), args.num_activities))
    message = f"Running {len(activities)} permutations of {args.num_activities} activities"
    print(message)

    with open(os.path.join(folder, "benchmark.log"), "a") as log_file:
        log_file.write(f"{message}\n\n")

    start = time.time()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=1) as pool:
        pool.map(run_benchmark, activities)

    end = time.time()
    message = f"Finished running {len(activities)} permutations in {end - start:.2f} seconds"
    print(message)
    
    with open(os.path.join(folder, "benchmark.log"), "a") as log_file:
        log_file.write(f"{message}\n")
