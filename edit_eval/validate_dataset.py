import sys
sys.path.append('.')

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

import numpy as np
import tqdm

from data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from execution import check_correctness

CODE_MARKER = r"{{Code}}"

def build_program_and_tests(problem, code):
    if "context" in problem.keys() and CODE_MARKER in problem["context"]:
        code = problem["context"].replace(CODE_MARKER, code)

    return (
        code + "\n\n" +
        problem["test"] + "\n\n" +
        # f"check({problem['entry_point']})"
        f"check()"
    )

######################################
# Methods for validating the dataset #
######################################

def assert_inputs_fail_tests(
    problem_file: str = None,
    n_workers: int = 4,
    timeout: float = 10.0,
):
    """
    Checks if all input programs (before edit) fails all the tests.
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        results = []

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(problem_file)):
            task_id = sample["task_id"]
            code = sample["input"]
            check_program = build_program_and_tests(sample, code)
            args = (problems[task_id], check_program, timeout, task_id)
            future = executor.submit(check_correctness, *args)
            futures.append(future)

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append((result["run_id"], result))

    all_correct = True
    for result in results:
        run_id, result = result
        passed = result["passed"]  # Only one code per task
        if passed:
            print(f"The input code of {run_id} should not pass the tests.")
            all_correct = False

    if all_correct:
        print("All input code (before edit) in the dataset failed as expected.")
    


def assert_target_pass_tests(
    problem_file: str = None,
    n_workers: int = 4,
    timeout: float = 10.0,
):
    """
    Checks if all input programs (before edit) fails all the tests.
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        results = []

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(problem_file)):
            task_id = sample["task_id"]
            code = sample["output"]
            check_program = build_program_and_tests(sample, code)
            args = (problems[task_id], check_program, timeout, task_id)
            future = executor.submit(check_correctness, *args)
            futures.append(future)

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results.append((result["run_id"], result))

    all_correct = True
    for result in results:
        run_id, result = result
        passed = result["passed"]  # Only one code per task
        if not passed:
            print(f"The target code of {run_id} should pass the tests.")
    
    if all_correct:
        print("All target code (after edit) in the dataset passed as expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validates the specified JSONL dataset.")
    parser.add_argument("file_path", type=str, help="Path to the JSONL file")
    
    args = parser.parse_args()

    assert_inputs_fail_tests(args.file_path)
    print("_" * 100)
    print("Testing target code")
    assert_target_pass_tests(args.file_path)