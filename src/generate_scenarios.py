import argparse
import json
import os
import random
import re
from collections import OrderedDict

import openai
import pandas as pd
import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential

from request_helper import ParallelRunner
from template.prompt_template import SCENARIO_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="temp/machine_generated_instructions.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="temp/machine_generated_scenarios.jsonl",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo",
        help="The engine to use."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="openai api key"
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        default='data/additional_seed.json',
        help="The path to the seed human-evaluated data.",
    )
    parser.add_argument(
        "--proc_num",
        type=int,
        default=2,
        help="The number of concurrent process."
    )

    parser.add_argument
    return parser.parse_args()


@retry(wait=wait_random_exponential(min=8, max=16), stop=stop_after_attempt(8))
def query_for_scenarios(item, key=None):
    if key is not None:
        openai.api_key = key
    
    task = item['instruction']
    prompt_str = item['prompt']

    response = openai.ChatCompletion.create(
        model = args.engine,
        messages = [
            {"role": "user", "content": prompt_str},
            {"role": "user", "content": f"Task: {task}"},
        ],
        temperature = 0.7,
        max_tokens = 1300,
        presence_penalty = 1.5,
    )

    scenario_list = response['choices'][0]['message']['content']
    scenario = ""
    if re.search("Scenario \d+:.+", scenario_list):
        scenarios = re.findall("Scenario \d+:.+", scenario_list)
        scenarios = [s.split(':')[1].strip() for s in scenarios]
        scenario = random.choice(scenarios)


    query_result = {
        'instruction': task,
        'scenario_list': scenario_list,
        'scenario': scenario,
        'response': response,
    }
    return query_result


def encode_prompt(seed_tasks):
    (shot1, shot2) = random.sample(seed_tasks, 2)
    # shot1
    shot1_scenarios = shot1['scenario_list'].strip()
    shot1_prompt = f"Task: {shot1['instruction']}\n{shot1_scenarios}"

    # shot2
    shot2_scenarios = shot2['scenario_list'].strip()
    shot2_prompt = f"Task: {shot2['instruction']}\n{shot2_scenarios}"

    few_shot_prompt = SCENARIO_PROMPT.format(SHOT1=shot1_prompt, SHOT2=shot2_prompt)
    return few_shot_prompt


def load_file(path):
    files = []
    if not os.path.exists(path):
        return files
    if path.endswith('.jsonl'):
        files = [json.loads(line) for line in open(path).readlines()]
    elif path.endswith('json'):
        files = json.loads(open(path).read())
    else:
        raise NotImplementedError
    return files


if __name__ == "__main__":
    args = parse_args()
    seed_tasks = load_file(args.seed_tasks_path)
    print(f"Total #seed tasks: {len(seed_tasks)}")

    tasks = load_file(args.input_file)
    print(f"Total #tasks: {len(tasks)}")
    
    exist_requests = set([item['instruction'] for item in load_file(args.output_file)])
    
    unresolved_tasks, resolved_tasks = [], []
    for task in tasks:
        hash = task['instruction']
        if hash in exist_requests:
            resolved_tasks.append(task)
        else:
            task['prompt'] = encode_prompt(seed_tasks)
            unresolved_tasks.append(task)
    
    with open(args.output_file, "a") as fout:
        runner = ParallelRunner(key=args.api_key, num_workers=args.proc_num, verbose=True)
        results = runner.start(data=unresolved_tasks,
                                query_func=query_for_scenarios, 
                                output_filename=args.output_file)