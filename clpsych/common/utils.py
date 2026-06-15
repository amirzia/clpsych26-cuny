import json
import re
from pathlib import Path

from clpsych.common.data_loader import Sequence


def convert_sequence_to_string(sequence: Sequence) -> str:
    posts_str = ''
    for i, post in enumerate(sequence.posts):
        content = post.content.replace('\n', ' ')
        posts_str += f"#### Post {i+1}\n{content}\n"
    return posts_str


def save_predictions(predicted_sequences: list[Sequence], result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    converted = [
        {
            'timeline_id': seq.timeline_id,
            'sequence_id': seq.sequence_id,
            'summary': seq.summary,
        }
        for seq in predicted_sequences
    ]
    with open(result_dir / 'predictions.json', 'w') as f:
        json.dump(converted, f, indent=4)


def extract_json_block(response: str) -> str:
    if '```json' in response:
        return re.findall(r'```json([\s\S]*?)```', response)[-1]
    return response
