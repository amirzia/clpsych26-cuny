

import json
import re
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
from pathlib import Path
import math

from enum import Enum

ELEMENT_CLASS_SIZE = [7, 2, 1, 2, 1, 3]
elements = ['A', 'B-O', 'B-S', 'C-O', 'C-S', 'D']
element_indices = {element: index for index, element in enumerate(elements)}

def get_subelement_index(subelement: str) -> int:
    index = int(re.match(r'\(([0-9]*)\)', subelement).group(1))
    return math.ceil(index / 2)

class StateType(Enum):
    adaptive = "adaptive"
    maladaptive = "maladaptive"

class ChangeType(Enum):
    switch = "Switch"
    escalation = "Escalation"

class StateElement(Enum):
    affect = 'A'
    behavior_others = 'B-O'
    behavior_self = 'B-S'
    cognition_others = 'C-O'
    cognition_self = 'C-S'
    desire = 'D'

@dataclass
class SelfState:
    post_id: str
    state_type: StateType
    evidence: str
    element: StateElement
    subelement: str

@dataclass
class Post:
    timeline_id: str
    post_id: str
    index: int
    content: str
    created_at: str
    adaptive_states: List[SelfState]
    adaptive_presence: int
    maladaptive_states: List[SelfState]
    maladaptive_presence: int
    wellbeing: int
    switch: bool
    escalation: bool

    def __str__(self) -> str:
        output = f'''Post(
    timeline_id={self.timeline_id},
    post_id={self.post_id},
    index={self.index},
    content={self.content},
    created_at={self.created_at},
    adaptive_states={self.adaptive_states},
    adaptive_presence={self.adaptive_presence},
    maladaptive_states={self.maladaptive_states},
    maladaptive_presence={self.maladaptive_presence},
    wellbeing={self.wellbeing},
    switch={self.switch},
    escalation={self.escalation}
)'''
        return output

    def get_state_vector(self) -> list:
        adaptive_state_vector = [0] * len(ELEMENT_CLASS_SIZE)
        maladaptive_state_vector = [0] * len(ELEMENT_CLASS_SIZE)
        for state in self.adaptive_states:
            element_index = element_indices[state.element.value]
            adaptive_state_vector[element_index] = get_subelement_index(state.subelement)
        for state in self.maladaptive_states:
            element_index = element_indices[state.element.value]
            maladaptive_state_vector[element_index] = get_subelement_index(state.subelement)
        return adaptive_state_vector + [self.adaptive_presence] + maladaptive_state_vector + [self.maladaptive_presence]

    def __repr__(self) -> str:
        return self.__str__()

@dataclass
class Timeline:
    timeline_id: str
    posts: List[Post]


@dataclass
class Sequence:
    timeline_id: str
    sequence_id: str
    posts: List[Post]
    change_type: ChangeType
    summary: str

train_users = [
    '306d938d4b',
    '83997cd4e7',
    '7d9d2e0e0a',
    '30c9e21337',
    '46f4bb3ada',
    '8080ac11fc',
    '8c5231d642',
    '9f2e135af8',
    'c62c65f481',
    '15564c5c62',
    'aa310342f5',
    'feb7900b62',
    'd0fb4b962e',
    '62defd8df2',
    '3db8573df5',
    'a35b41d1ca',
    '8e0a58cfbd',
    'caee9a5e58',
    'cf9aadf596',
    '0cac13e357',
]

val_users = [
    'b0185a6f5d',
    '43f395b896',
    'd237ea4269',
    '6c9677b482',
    '5da839acb5',
    '8fc3f6c07d',
    '91b6a42835',
    'a2aa387774',
    '87821f81b9',
    'b1d762ee9f'
]

def load_timeline(timeline_raw: dict) -> Timeline:
    posts = []
    for post_raw in timeline_raw['posts']:
        post_index = post_raw.get('post_index', None)
        post_id = post_raw['post_id']
        date = post_raw.get('date', None)
        # switch = post_raw.get('Switch', '0') != '0'
        switch = None if 'Switch' not in post_raw else post_raw['Switch'] != '0'
        # escalation = post_raw.get('Escalation', '0') != '0'
        escalation = None if 'Escalation' not in post_raw else post_raw['Escalation'] != '0'
        # wellbeing = post_raw.get('Well-being', 1)
        wellbeing = None if 'Well-being' not in post_raw else post_raw['Well-being']

        def load_states(state_type: StateType) -> tuple[List[SelfState], int]:
            presence = None
            states = []
            for evidence_type, evidence in post_raw['evidence'][f'{state_type.value}-state'].items():
                if evidence_type == 'Presence':
                    presence = evidence
                    continue
                element = StateElement(evidence_type)
                state = SelfState(
                    post_id=post_id,
                    state_type=state_type,
                    evidence=evidence['highlighted_evidence'],
                    element=element,
                    subelement=evidence['Category']
                )
                states.append(state)
            if not presence:
                presence = 1
            return states, presence

        if 'evidence' in post_raw:
            adaptive_states, adaptive_presence = load_states(StateType.adaptive)
            maladaptive_states, maladaptive_presence = load_states(StateType.maladaptive)
        else:
            adaptive_states = None
            maladaptive_states = None
            adaptive_presence = None
            maladaptive_presence = None

        post = Post(
            timeline_id=timeline_raw['timeline_id'],
            post_id=post_id,
            index=post_index,
            content=post_raw['post'],
            created_at=date,
            adaptive_states=adaptive_states,
            adaptive_presence=adaptive_presence,
            maladaptive_states=maladaptive_states,
            maladaptive_presence=maladaptive_presence,
            wellbeing=wellbeing,
            switch=switch,
            escalation=escalation
        )
        posts.append(post)
    return Timeline(
        timeline_id=timeline_raw['timeline_id'],
        posts=posts
    )


def read_data_train(data_path: str | Path) -> tuple[List[Timeline], List[Timeline]]:
    '''Read the data from the data path and split it into train and val sets'''
    if isinstance(data_path, str):
        data_path = Path(data_path)
    train_timelines = []
    val_timelines = []
    for filename in data_path.glob('*.json'):
        with open(filename, 'r') as f:
            timeline_raw = json.load(f)
            timeline = load_timeline(timeline_raw)
            if filename.stem in train_users:
                # timeline.posts = [post for post in timeline.posts if post.adaptive_presence > 1 or post.maladaptive_presence > 1]
                train_timelines.append(timeline)
            else:
                val_timelines.append(timeline)
    return train_timelines, val_timelines

def read_data_train_test(train_path: str | Path, test_path: str | Path) -> tuple[List[Timeline], List[Timeline]]:
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_timelines = []
    test_timelines = []
    for filename in train_path.glob('*.json'):
        with open(filename, 'r') as f:
            timeline_raw = json.load(f)
            timeline = load_timeline(timeline_raw)
            train_timelines.append(timeline)

    for filename in test_path.glob('*.json'):
        with open(filename, 'r') as f:
            timeline_raw = json.load(f)
            timeline = load_timeline(timeline_raw)
            test_timelines.append(timeline)

    return train_timelines, test_timelines

def read_sequence_data(timelines: List[Timeline], data_path: str | Path) -> List[Sequence]:
    data = json.load(open(data_path))
    sequences = []
    for entry in data:
        timeline_id = entry['timeline_id']
        sequence_id = entry['sequence_id']
        posts = []
        timeline = [t for t in timelines if t.timeline_id == timeline_id][0]
        for post_id, post_index in zip(entry['postids'], entry['postindices']):
            post_index -= 1
            if timeline.posts[post_index].post_id != post_id:
                raise ValueError(f'Post ID mismatch: post_index={post_index}, post_id1={timeline.posts[post_index].post_id}, post_id2={post_id}')
            posts.append(timeline.posts[post_index])
        change_type = ChangeType(entry['change_type']) if 'change_type' in entry else None
        summary = entry.get('summary', None)
        sequence = Sequence(
            timeline_id=timeline_id,
            sequence_id=sequence_id,
            posts=posts,
            change_type=change_type,
            summary=summary
        )
        sequences.append(sequence)
    return sequences

def load_timelines(
    train_dir: str | Path,
    test_dir: str | Path | None = None,
    *,
    test: bool = False,
) -> tuple[List[Timeline], List[Timeline]]:
    '''Load timelines split into (train, eval) consistently across tasks.

    Validation mode (``test=False``): every timeline is read from ``train_dir``
    and partitioned into the ``train_users`` / ``val_users`` split; ``eval`` is
    the validation split.

    Test mode (``test=True``): ``train`` is every timeline in ``train_dir`` and
    ``eval`` is every timeline in ``test_dir``.
    '''
    if test:
        if test_dir is None:
            raise ValueError('test_dir is required in test mode')
        return read_data_train_test(train_dir, test_dir)
    return read_data_train(train_dir)


def load_sequences(
    train_dir: str | Path,
    train_sequences_file: str | Path,
    test_dir: str | Path | None = None,
    test_sequences_file: str | Path | None = None,
    *,
    test: bool = False,
) -> tuple[List[Sequence], List[Sequence]]:
    '''Load Task 3 sequences split into (train, eval), matching ``load_timelines``.

    Validation mode (``test=False``): all timelines from ``train_dir`` are loaded
    so both splits' sequences resolve, then sequences are partitioned by the
    ``train_users`` / ``val_users`` split; ``eval`` is the validation split.

    Test mode (``test=True``): ``train`` sequences come from ``train_dir`` /
    ``train_sequences_file`` and ``eval`` sequences from ``test_dir`` /
    ``test_sequences_file``.
    '''
    if test:
        if test_dir is None or test_sequences_file is None:
            raise ValueError('test_dir and test_sequences_file are required in test mode')
        train_timelines, test_timelines = read_data_train_test(train_dir, test_dir)
        train_sequences = read_sequence_data(train_timelines, train_sequences_file)
        eval_sequences = read_sequence_data(test_timelines, test_sequences_file)
        return train_sequences, eval_sequences

    train_timelines, val_timelines = read_data_train(train_dir)
    all_sequences = read_sequence_data(train_timelines + val_timelines, train_sequences_file)
    train_sequences = [s for s in all_sequences if s.timeline_id in train_users]
    eval_sequences = [s for s in all_sequences if s.timeline_id in val_users]
    return train_sequences, eval_sequences


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=Path, required=True)
    p.add_argument('--sequences_file', type=Path, required=True)
    a = p.parse_args()
    train_timelines, val_timelines = read_data_train(a.train_dir)
    all_timelines = train_timelines + val_timelines
    sequences = read_sequence_data(all_timelines, a.sequences_file)
    print(f'train={len(train_timelines)}  val={len(val_timelines)}  sequences={len(sequences)}')