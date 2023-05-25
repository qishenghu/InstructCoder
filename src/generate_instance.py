import argparse
import json
import os
import random
import re
from collections import OrderedDict

import openai
import pandas as pd
import tiktoken
import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential

from request_helper import ParallelRunner
from template.prompt_template import INSTANCE_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="temp/machine_generated_scenarios.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="temp/machine_generated_instances.jsonl",
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
def query_for_instance(item, key=None):
    if key is not None:
        openai.api_key = key
    
    task = item['instruction']
    scenario = item['scenario']
    scenario_list = item['scenario_list']
    prompt_str = item['prompt']

    response = openai.ChatCompletion.create(
        model = args.engine,
        messages = [
            {"role": "user", "content": prompt_str},
        ],
        temperature = 0.,
        max_tokens = 1024,
        presence_penalty = 1.5,
    )

    output = response['choices'][0]['message']['content']
    query_result = {
        'instruction': task,
        'scenario': scenario,
        'scenario_list': scenario_list,
        'result': output,
        'response': response
    }
    return query_result


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


def encode_prompt(item, tasks):
    start_prompt = INSTANCE_PROMPT
    shots = random.sample(tasks, 4)
    for n_shot in range(4, 0, -1):
        prompt = start_prompt
        for shot in shots[:n_shot]:
            prompt += f"Scenario: {shot['scenario']}\nTask: {shot['instruction']}\n"
            prompt += f"Input: \n{shot['input']}\n\nOutput: \n{shot['output']}\n\n\n"
        prompt += f"Scenario: {item['scenario']}\nTask: {item['instruction']}\n"
        if len(tokenizer.encode(prompt)) < 3072:
            return prompt
    return start_prompt


if __name__ == "__main__":
    args = parse_args()
    tokenizer = tiktoken.encoding_for_model("gpt2")
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
            task['prompt'] = encode_prompt(task, seed_tasks)
            unresolved_tasks.append(task)
    
    with open(args.output_file, "a") as fout:
        runner = ParallelRunner(key=args.api_key, num_workers=args.proc_num, verbose=True)
        results = runner.start(data=unresolved_tasks,
                                query_func=query_for_instance, 
                                output_filename=args.output_file)