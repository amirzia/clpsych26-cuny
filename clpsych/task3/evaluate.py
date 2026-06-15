prompt = '''You are an expert in psychology. Your task is to evaluate a structured summary \
describing patterns of self-state dynamics and their progression over time within a sequence of Reddit posts \
You are given the gold summary and the predicted summary.

Your task is to evaluate the quality of the generated \
summary vs. the gold summary in terms of consistency and contradiction. The consistency is the degree to which the \
predicted summary is consistent with the gold summary and is floated point number ranged from 0 (completely inconsistent) to \
1 (completely consistent). The contradiction is the degree to which the predicted summary \
is contradictory to the gold summary and is floated point number ranged from 0 (no contradiction) to 1 (completely contradictory).

The output should be a JSON object with the following format:
```json
{
    "consistency": float,
    "contradiction": float
}
```
'''

import argparse
from pathlib import Path
import json

from openai import OpenAI

from clpsych.common.llm import query_llm
from clpsych.common.data_loader import (
    load_sequences,
)
from clpsych.common.utils import extract_json_block


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-27B')
    parser.add_argument('--api_address', type=str, default='http://127.0.0.1:8000/v1')
    parser.add_argument('--prediction_file', type=Path, required=True)
    parser.add_argument('--train_dir', type=Path)
    parser.add_argument('--train_sequences_file', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI(api_key='None', base_url=args.api_address)
    prediction_file = Path(args.prediction_file)
    prediction_dir = prediction_file.parent

    _, validation_sequences = load_sequences(args.train_dir, args.train_sequences_file)

    results = []
    predictions = json.load(open(prediction_file))
    for prediction in predictions:
        gold_sequence = [
            s for s in validation_sequences
            if s.timeline_id == prediction['timeline_id'] and s.sequence_id == prediction['sequence_id']
        ][0]
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f'Predicted summary: {prediction["summary"]}\n\nGold summary: {gold_sequence.summary}'}
        ]
        response = query_llm(client, args.model, messages)
        response = json.loads(extract_json_block(response))
        results.append({
            'timeline_id': prediction['timeline_id'],
            'sequence_id': prediction['sequence_id'],
            'consistency': response['consistency'],
            'contradiction': response['contradiction'],
        })

    with open(prediction_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
