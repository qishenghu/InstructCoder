import argparse
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import openai
import pandas as pd
import tqdm
from rouge_score import rouge_scorer
from tenacity import retry, stop_after_attempt, wait_random_exponential

from request_helper import ParallelRunner
from template.prompt_template import EDIT_TYPES, INSTRUCTION_PROMPT, INSTRUCTION_PROMPT_WITH_TYPE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./temp/",
        help="The directory where the generated file will be saved",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        nargs="+",
        default=['./data/github_seed.json', './data/additional_seed.json'],
        help="The path to the seed human-evaluated data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=5,
        help="th",
    )
    parser.add_argument(
        "--use_edit_type",
        action="store_true",
        help="If specified, will use template with edit type for generating instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="openai api key"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The number of concurrent process."
    )
    parser.add_argument(
        "--proc_num",
        type=int,
        default=2,
        help="The number of concurrent process."
    )

    return parser.parse_args()


def sample_items(total, n):
    """Sample n items from a list of total instructions."""
    samples = []
    if len(total) == 0 or n == 0:
        return samples
    samples = random.sample(total, min(n, len(total)))
    return samples


@retry(wait=wait_random_exponential(min=6, max=16), stop=stop_after_attempt(1))
def query_for_instruction(items, key):
    item = items if isinstance(items, dict) else items[0]
    user_message_1, user_message_2 = item['user_message_1'], item['user_message_2']
    response = openai.ChatCompletion.create(
        api_key = key,
        model = args.engine,
        messages = [
            {"role": "system", "content": "You are an experienced python developer."},
            {"role": "user", "content": user_message_1},
            {"role": "user", "content": user_message_2},
        ],
        temperature = 0.7,
        top_p=0.5,
        presence_penalty=1.5,
        max_tokens = 2048,
    )
    output = response['choices'][0]['message']['content']
    pt = re.compile("(?<=\").+(?=\")")
    if pt.search(output):
        instructions = pt.findall(output)
        query_results = []
        for instruction in instructions:
            query_result = {
                'instruction': instruction,
            }
            query_results.append(query_result)
        return query_results
    else:
        instruction = "None"
    query_result = {
        'instruction': instruction,
    }
    return [query_result]
    

def get_rouge_scores(inst, all_instructions):
    if inst in all_instructions:
        all_instructions.remove(inst)
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    for e_inst in all_instructions:
        rouge_scores.append(scorer.score(inst, e_inst))
    rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
    max_score = max(rouge_scores)
    most_similar_instructions = {
        all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
    }
    mean_score = float(np.mean(rouge_scores))
    all_instructions.append(inst)
    return max_score, most_similar_instructions, mean_score


def encode_conversation(instructions, edit_type=''):
    if edit_type:
        user_message_1 = INSTRUCTION_PROMPT_WITH_TYPE.format(edit_task=edit_type)
    else:
        user_message_1 = INSTRUCTION_PROMPT
    user_message_2 = ""
    for i in range(len(instructions)):
        user_message_2 += f"{i+1}. \"{instructions[i]}\"\n"
    return user_message_1, user_message_2


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
    random.seed(args.seed)
    # Seed task loading
    seed_tasks = []
    print(args.seed_tasks_path)
    for path in args.seed_tasks_path:
        seed_tasks.extend(load_file(path))
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print("Loaded {} seed tasks.".format(len(seed_instructions)))
    os.makedirs(args.save_dir, exist_ok=True)
    batch_size = args.proc_num

    # LM-generated task loading
    machine_tasks_path = os.path.join(args.save_dir, "machine_generated_instructions.jsonl")
    machine_tasks = load_file(machine_tasks_path)
    machine_instructions = [row['instruction'] for row in machine_tasks]
    print(f"Found {len(machine_instructions)} machine-generated instructions.")
    
    # Intermediate output path
    inter_inst_path = os.path.join(args.save_dir, "inter_instructions.jsonl")
    
    exist_num = len(machine_tasks)
    print(f"Generate {args.num_instructions_to_generate} instructions in total. \
          Found {exist_num} existing instructions. \
          Still need to generate {args.num_instructions_to_generate - exist_num} instructions.")
    
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(exist_num)
    
    edit_type_idx = 0
    while len(machine_instructions) < args.num_instructions_to_generate:
        batch_inputs = []
        for idx in range(min(batch_size, args.num_instructions_to_generate - len(machine_instructions))):
            # Select an edit_type
            edit_type = None
            if args.use_edit_type:
                edit_type = EDIT_TYPES[edit_type_idx % len(EDIT_TYPES)]
                edit_type_idx += 1

            # Sample batch instructions from the pool
            prompt_tasks = []
            if machine_instructions:
                prompt_tasks = sample_items(machine_instructions, n=1)
            prompt_tasks += sample_items(seed_instructions, args.num_prompt_instructions - len(prompt_tasks))
            random.shuffle(prompt_tasks)

            # Encode the conversation
            user_message_1, user_message_2 = encode_conversation(prompt_tasks, edit_type=edit_type)
            batch_inputs.append({'user_message_1': user_message_1, 'user_message_2': user_message_2})

        # Query to generate instruction
        runner = ParallelRunner(key=args.api_key, num_workers=args.proc_num, verbose=False)
        results = runner.start(data=batch_inputs,
                                query_func=query_for_instruction, 
                                output_filename=inter_inst_path,
                                batch=True)
    
        # Filter and write results
        instructions = [result['instruction'] for result in results]
        all_instructions = seed_instructions + machine_instructions + instructions
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        for inst in instructions:
            with Pool(4) as p:
                rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
            rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
            if max(rouge_scores) > 0.7:
                continue
            failure_indicators = [
                'no new task',
                'no task instruction',
                'no change',
                'no new instruction'
                'no new task',
                'sorry',
            ]
            if any([phrase in inst.lower() for phrase in failure_indicators]):
                continue
            
            machine_instructions.append(inst)
            with open(machine_tasks_path, 'a') as f:
                f.write(json.dumps({'instruction': inst}) + '\n')
            progress_bar.update(1)

            if len(machine_instructions) >= args.num_instructions_to_generate:
                break
            
    
    print(f"Saved {len(machine_instructions)} instructions.")