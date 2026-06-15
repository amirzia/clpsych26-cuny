import argparse
from pathlib import Path
from copy import deepcopy
import json
import time

from openai import OpenAI

from clpsych.task3.prompts import JUDGE_AGGREGATE_PROMPT
from clpsych.common.llm import query_llm
from clpsych.common.data_loader import (
    Sequence,
    load_sequences,
)
from clpsych.common.utils import convert_sequence_to_string, save_predictions, extract_json_block


def make_prompt(sequence: Sequence, example_summaries: list[str]) -> list[dict[str, str]]:
    posts_str = convert_sequence_to_string(sequence)
    header = '# Example summaries'
    examples_str = ''
    for i, summary in enumerate(example_summaries):
        examples_str += f'## Example summary {i+1}\n'
        examples_str += f'{summary}\n'

    user_content = header + '\n' + 'Post: \n' + posts_str + '\n' + examples_str
    return [
        {'role': 'system', 'content': JUDGE_AGGREGATE_PROMPT},
        {'role': 'user', 'content': user_content}
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-27B')
    parser.add_argument('--api_address', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--result_dir', type=Path, required=True)
    parser.add_argument('--example_summaries_files', type=str, required=True, nargs='+')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_dir', type=Path)
    parser.add_argument('--test_dir', type=Path)
    parser.add_argument('--train_sequences_file', type=Path)
    parser.add_argument('--test_sequences_file', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI(api_key='None', base_url=args.api_address)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    _, test_sequences = load_sequences(
        args.train_dir, args.train_sequences_file,
        args.test_dir, args.test_sequences_file,
        test=args.test,
    )

    predicted_summaries = []
    for example_summaries_file in args.example_summaries_files:
        with open(example_summaries_file, 'r') as f:
            predicted_summaries.append(json.load(f))

    predicted_sequences = []
    for seq_idx in range(len(predicted_summaries[0])):
        timeline_id = test_sequences[seq_idx].timeline_id
        sequence_id = test_sequences[seq_idx].sequence_id
        for pred_idx in range(len(predicted_summaries)):
            assert predicted_summaries[pred_idx][seq_idx]['timeline_id'] == timeline_id
            assert predicted_summaries[pred_idx][seq_idx]['sequence_id'] == sequence_id
        sample_summaries = [predicted_summaries[pred_idx][seq_idx]['summary'] for pred_idx in range(len(predicted_summaries))]
        sequence = test_sequences[seq_idx]
        messages = make_prompt(sequence, sample_summaries)
        if args.debug:
            print("Prompt: ", messages[0]['content'] + '\n\n' + messages[1]['content'])
            print("Response: ")
        while True:
            try:
                response = query_llm(client, args.model, messages, print_output=args.debug)
                response = json.loads(extract_json_block(response))
                pred_seq = deepcopy(sequence)
                pred_seq.summary = response['summary']
                predicted_sequences.append(pred_seq)
                break
            except Exception as e:
                print(f"Error: {e}. Retrying...")
                time.sleep(2)

    save_predictions(predicted_sequences, result_dir)


if __name__ == '__main__':
    main()
