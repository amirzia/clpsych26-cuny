
from pathlib import Path
import json

import argparse
import pandas as pd
import numpy as np

from clpsych.common.data_loader import read_data_train


def get_post2timeline(data_path: Path) -> dict[str, str]:
    train_timelines, val_timelines = read_data_train(data_path)
    post2timeline = {}
    for timeline in train_timelines + val_timelines:
        for post in timeline.posts:
            post2timeline[post.post_id] = timeline.timeline_id
    return post2timeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=Path, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--train_dir', type=Path, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    data = pd.read_csv(args.input_path)
    post2timeline = get_post2timeline(args.train_dir)
    results = []
    for _, row in data.iterrows():
        post_id = row['post_id']
        if 'timeline_id' not in row:
            timeline_id = post2timeline[post_id]
        else:
            timeline_id = row['timeline_id']
        entry = {
            'timeline_id': timeline_id,
            'post_id': post_id,
            'adaptive-state': {},
            'maladaptive-state': {},
        }

        a_adaptive = min(row['llm_adaptive_1'] * 2 - 1, 13)
        bo_adaptive = min(row['llm_adaptive_2'] * 2 - 1, 3)
        bs_adaptive = min(row['llm_adaptive_3'] * 2 - 1, 1)
        co_adaptive = min(row['llm_adaptive_4'] * 2 - 1, 3)
        cs_adaptive = min(row['llm_adaptive_5'] * 2 - 1, 1)
        d_adaptive = min(row['llm_adaptive_6'] * 2 - 1, 5)
        presence_adaptive = row['llm_adaptive_presence']

        a_maladaptive = min(row['llm_maladaptive_1'] * 2, 14)
        bo_maladaptive = min(row['llm_maladaptive_2'] * 2, 4)
        bs_maladaptive = min(row['llm_maladaptive_3'] * 2, 2)
        co_maladaptive = min(row['llm_maladaptive_4'] * 2, 4)
        cs_maladaptive = min(row['llm_maladaptive_5'] * 2, 2)
        d_maladaptive = min(row['llm_maladaptive_6'] * 2, 6)
        presence_maladaptive = row['llm_maladaptive_presence']

        if a_adaptive > 0:
            entry['adaptive-state']['A'] = {
                'subelement': a_adaptive,
            }
        if bo_adaptive > 0:
            entry['adaptive-state']['B-O'] = {
                'subelement': bo_adaptive,
            }
        if bs_adaptive > 0:
            entry['adaptive-state']['B-S'] = {
                'subelement': bs_adaptive,
            }
        if co_adaptive > 0:
            entry['adaptive-state']['C-O'] = {
                'subelement': co_adaptive,
            }
        if cs_adaptive > 0:
            entry['adaptive-state']['C-S'] = {
                'subelement': cs_adaptive,
            }
        if d_adaptive > 0:
            entry['adaptive-state']['D'] = {
                'subelement': d_adaptive,
            }
        if presence_adaptive > 0:
            entry['adaptive-state']['Presence'] = presence_adaptive

        if a_maladaptive > 0:
            entry['maladaptive-state']['A'] = {
                'subelement': a_maladaptive,
            }
        if bo_maladaptive > 0:
            entry['maladaptive-state']['B-O'] = {
                'subelement': bo_maladaptive,
            }
        if bs_maladaptive > 0:
            entry['maladaptive-state']['B-S'] = {
                'subelement': bs_maladaptive,
            }
        if co_maladaptive > 0:
            entry['maladaptive-state']['C-O'] = {
                'subelement': co_maladaptive,
            }
        if cs_maladaptive > 0:
            entry['maladaptive-state']['C-S'] = {
                'subelement': cs_maladaptive,
            }
        if d_maladaptive > 0:
            entry['maladaptive-state']['D'] = {
                'subelement': d_maladaptive,
            }
        if presence_maladaptive > 0:
            entry['maladaptive-state']['Presence'] = presence_maladaptive

        results.append(entry)

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
