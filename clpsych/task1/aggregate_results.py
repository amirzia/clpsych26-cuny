import argparse
from pathlib import Path
import json
from collections import Counter

import numpy as np

subelements = ['A', 'B-O', 'B-S', 'C-O', 'C-S', 'D']
valences = ['adaptive-state', 'maladaptive-state']


def aggregate_results(results: list[list[dict]]) -> list[dict]:
    """Combine several Task 1 submissions into one by majority vote.

    Each input is a parsed submission: a list of per-post prediction dicts, all
    covering the same posts in the same order. For every post and valence, each
    subelement index is chosen by majority vote across the inputs and the
    presence rating is the (rounded) median.
    """
    n_posts = len(results[0])
    for other in results[1:]:
        if len(other) != n_posts:
            raise ValueError(f'Submissions have different lengths: {n_posts} vs {len(other)}')

    combined = []
    for index in range(n_posts):
        post_id = results[0][index]['post_id']
        for other in results[1:]:
            if other[index]['post_id'] != post_id:
                raise ValueError(
                    f'Post mismatch at index {index}: {post_id} vs {other[index]["post_id"]}'
                )

        entry = {
            'timeline_id': results[0][index]['timeline_id'],
            'post_id': post_id,
            'adaptive-state': {},
            'maladaptive-state': {},
        }
        for valence in valences:
            for subelement in subelements:
                subelement_values = [res[index][valence].get(subelement, {}).get('subelement', 0) for res in results]
                most_common = Counter(subelement_values).most_common(1)[0][0]
                if most_common != 0:
                    entry[valence][subelement] = {
                        'subelement': most_common,
                    }

            presences = [res[index][valence]['Presence'] for res in results]
            presence = int(np.round(np.median(presences)))
            entry[valence]['Presence'] = presence
        combined.append(entry)
    return combined


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ensemble Task 1 submissions by majority vote on subelements and median presence.'
    )
    parser.add_argument(
        '--prediction_files', type=Path, nargs='+', required=True,
        help='Two or more Task 1 submission JSON files to ensemble (the ensemble members). '
             'All must cover the same posts in the same order.',
    )
    parser.add_argument(
        '--output_path', type=Path, required=True,
        help='Path to write the combined submission JSON.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if len(args.prediction_files) < 2:
        raise SystemExit('Provide at least two --prediction_files to ensemble.')
    results = [json.loads(path.read_text()) for path in args.prediction_files]
    combined = aggregate_results(results)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(combined, f, indent=4)
    print(f'Ensembled {len(results)} submissions into {len(combined)} posts -> {args.output_path}')
