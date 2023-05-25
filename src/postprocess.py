import argparse
import json
import os
import random
import string

from datasketch import MinHash, MinHashLSH


def format_instance(res):
    input, output = '', ''
    if 'Input:' in res and 'Output:' in res:
        input = res.split('Output:')[0].split('Input:')[1].strip('\n')
        output = res.split('Output:')[1].strip('\n')
        if input.startswith(' \n'):
            input = input.lstrip(' \n')
        if output.startswith(' \n'):
            output = output.lstrip(' \n')
    return input, output


def format_filter(input, output):
    failure_indicators = ['sorry', 'please provide', 'cannot', 'not able', 'already']
    if input == "" or output == "":
        return False
    if input == output:
        return False
    if input.strip().endswith(":") or output.strip().endswith(":"):
        return False
    if any([phrase in input.lower() for phrase in failure_indicators]):
        return False
    if any([phrase in output.lower() for phrase in failure_indicators]):
        return False
    return True


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


def filter_duplicate_instance(instances, threshold=0.75):
    unique_instances = []
    lsh = MinHashLSH(threshold=threshold)
    for idx, instance in enumerate(instances):
        instance_string = instance['input']
        instance_string = instance_string.translate(str.maketrans('', '', string.punctuation))
        instance_string = instance_string.strip().split()
        mh = MinHash()
        for d in instance_string:
            mh.update(d.encode('utf8'))
        if not lsh.is_empty():
            res = lsh.query(mh)
            if len(res) != 0:
                continue
        lsh.insert(str(idx), mh)
        unique_instances.append(instance)
    del lsh
    return unique_instances
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="temp/machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="temp/instances.jsonl",
    )

    parser.add_argument
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    machine_generated_instances = load_file(args.input_file)
    print(f"Total #machine_generated_instances: {len(machine_generated_instances)}")
    formatted_instances = []
    for instance in machine_generated_instances:
        instance['input'], instance['output'] = format_instance(instance['result'])
        if format_filter(instance['input'], instance['output']):
            formatted_instances.append(instance)
    print(f"Total #formatted_instances: {len(formatted_instances)}")

    unique_instances = filter_duplicate_instance(formatted_instances)
    print(f"Total #unique_instances: {len(unique_instances)}")
    
    with open(args.output_file, 'w') as f:
        for instance in unique_instances:
            f.write(json.dumps({
                'instruction': instance['instruction'],
                'input': instance['input'],
                'output': instance['output'],
                'scenario': instance['scenario'],
                'scenario_list': instance['scenario_list'],
            }) + '\n')