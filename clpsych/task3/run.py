import argparse
from pathlib import Path
from copy import deepcopy
import random
import json
import re
import time

from openai import OpenAI

from clpsych.task3.prompts import FEW_SHOT
from clpsych.common.llm import query_llm
from clpsych.common.data_loader import (
    Sequence,
    load_sequences,
)
from clpsych.common.utils import convert_sequence_to_string, save_predictions, extract_json_block


def make_prompt(sequence: Sequence, train_sequences: list[Sequence], config: dict) -> list[dict[str, str]]:
    posts_str = convert_sequence_to_string(sequence)
    if config['mode'] == 'zero_shot':
        return [
            {'role': 'system', 'content': FEW_SHOT},
            {'role': 'user', 'content': f'Posts: \n\n {posts_str}'}
        ]

    samples = random.sample(train_sequences, config['k'])
    header = '# Examples'
    examples_str = ''
    for i, sample in enumerate(samples):
        examples_str += f'## Example {i+1}\n'
        examples_str += f"### Posts\n"
        examples_str += f'{convert_sequence_to_string(sample)}'
        examples_str += f"### Summary\n"
        examples_str += f'{sample.summary}\n'

    user_content = header + '\n' + examples_str + '\n\n' + 'Posts: \n' + posts_str
    return [
        {'role': 'system', 'content': FEW_SHOT},
        {'role': 'user', 'content': user_content}
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-27B')
    parser.add_argument('--api_address', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--mode', type=str, choices=['zero_shot', 'few_shot'], default='zero_shot')
    parser.add_argument('--result_dir', type=Path, required=True)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_dir', type=Path)
    parser.add_argument('--test_dir', type=Path)
    parser.add_argument('--train_sequences_file', type=Path)
    parser.add_argument('--test_sequences_file', type=Path)

    args = parser.parse_args()

    if not args.test and not (args.train_dir or args.train_sequences_file):
        print("Error: --train_dir or --train_sequences_file is required when --test is False")
        exit(1)

    if args.test and not (args.test_dir or args.test_sequences_file):
        print("Error: --test_dir or --test_sequences_file is required when --test is True")
        exit(1)


    return args


def main():
    args = parse_args()
    client = OpenAI(api_key='None', base_url=args.api_address)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'mode': args.mode,
        'k': args.k,
    }

    train_sequences, test_sequences = load_sequences(
        args.train_dir, args.train_sequences_file,
        args.test_dir, args.test_sequences_file,
        test=args.test,
    )

    predicted_sequences = []
    for sequence in test_sequences:
        messages = make_prompt(sequence, train_sequences, config)
        if args.debug:
            print("Prompt: ", messages[0]['content'] + '\n\n' + messages[1]['content'])
            print("Response: ")
        while True:
            try:
                response = query_llm(client, args.model, messages, print_output=args.debug)
                response = json.loads(extract_json_block(response))
                break
            except Exception as e:
                print(f"Error: {e}. Retrying...")
                time.sleep(2)
        pred_seq = deepcopy(sequence)
        pred_seq.summary = response['summary']
        predicted_sequences.append(pred_seq)

    save_predictions(predicted_sequences, result_dir)


if __name__ == '__main__':
    main()
