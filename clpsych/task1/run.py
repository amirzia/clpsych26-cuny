import argparse
import json
from pathlib import Path
import random
import re
from datetime import datetime
import logging
from typing import Any
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from openai import OpenAI

from clpsych.task1.prompts import (
    ZERO_SHOT,
    FEW_SHOT_SUBELEMENT_EXAMPLES,
    FEW_SHOT_POST_EXAMPLES,
    POST_EXAMPLES
)

from clpsych.common.llm import query_llm

from clpsych.common.data_loader import (
    Timeline,
    Post,
    ELEMENT_CLASS_SIZE,
    load_timelines,
    get_subelement_index,
)

pre_computed_embeddings: dict[str, Any] = {}

def save_embeddings(timelines: list[Timeline]) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'BAAI/bge-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    sentences = [post.content for timeline in timelines for post in timeline.posts]
    post_ids = [post.post_id for timeline in timelines for post in timeline.posts]
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    encoded_input = encoded_input.to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
        sentence_embeddings = outputs[0][:, 0]

    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    sentence_embeddings = sentence_embeddings.detach().cpu().numpy()

    for post_id, embedding in zip(post_ids, sentence_embeddings):
        pre_computed_embeddings[post_id] = embedding

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

elements = ['A', 'B-O', 'B-S', 'C-O', 'C-S', 'D']

def parse_response(response: str) -> list[int]:
    if 'json' in response:
        try:
            response = re.findall(r'```json([\s\S]*?)```', response)[-1]
        except Exception as e:
            print(f"Error in parsing the response: {e}")
            print(response)
            return None
    response = '\n'.join([line.strip().split('#')[0] for line in response.split('\n')])
    response = json.loads(response)
    adaptive_states = [response['adaptive_states'][element] for element in elements]
    adaptive_presence = response['adaptive_states']['rating']
    maladaptive_states = [response['maladaptive_states'][element] for element in elements]
    maladaptive_presence = response['maladaptive_states']['rating']

    if sum(adaptive_states) == 0:
        adaptive_presence = 1
    if sum(maladaptive_states) == 0:
        maladaptive_presence = 1
    return adaptive_states + [adaptive_presence] + maladaptive_states + [maladaptive_presence]


def sample_post_evidences(train_dataset: list[Timeline], k: int, sampling_type: str, post: Post) -> dict[str, list[str]]:

    train_posts = [post for timeline in train_dataset for post in timeline.posts]
    if sampling_type == 'random':
        sampled_posts = random.sample(train_posts, k)
        return sampled_posts
    elif sampling_type == 'embedding':
        embedding = pre_computed_embeddings[post.post_id]
        # use cosine similarity to find the most similar posts
        cosine_similarities = [np.dot(embedding, pre_computed_embeddings[other_post.post_id]) / (np.linalg.norm(embedding) * np.linalg.norm(pre_computed_embeddings[other_post.post_id])) for other_post in train_posts]
        sampled_posts = [train_posts[i] for i in np.argsort(cosine_similarities)][-k:]
        return sampled_posts

def make_prompt(post: Post, train_dataset: list[Timeline], config: dict[str, Any]) -> list[dict[str, str]]:
    if config['mode'] == 'zero_shot':
        return [
            {'role': 'system', 'content': ZERO_SHOT},
            {'role': 'user', 'content': f'Post: {post.content}'}
        ]
    assert config['mode'] == 'few_shot'
    if config['example_level'] == 'subelement':
        train_evidence = get_subelement_evidences(train_dataset)
        samples = sample_subelement_evidences(train_evidence, config['k'])
        system_prompt = FEW_SHOT_SUBELEMENT_EXAMPLES
        mapping = {}
        for placeholder, evidences in samples.items():
            if evidences:
                evidences = [evidence.replace('"', '\'').replace('\n', ' ') for evidence in evidences]
                evidence_str = '"'.join(evidences)
                evidence_str = f'"{evidence_str}"'
            else:
                evidence_str = 'None'
            mapping[placeholder] = evidence_str
        system_prompt = system_prompt.substitute(**mapping)
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Post: {post.content}'}
        ]
    elif config['example_level'] == 'post':
        samples = sample_post_evidences(train_dataset, config['k'], config['sampling_type'], post)
        in_context_str = ''
        for sample in samples:
            mapping = {
                'post_content': sample.content,

                'ad_A': '0',
                'ad_BO': '0',
                'ad_BS': '0',
                'ad_CO': '0',
                'ad_CS': '0',
                'ad_D': '0',
                'ad_rating': '0',

                'ma_A': '0',
                'ma_BO': '0',
                'ma_BS': '0',
                'ma_CO': '0',
                'ma_CS': '0',
                'ma_D': '0',
                'ma_rating': '0',
            }

            for state in sample.adaptive_states:
                element = state.element.value
                evidence = state.evidence
                subelement_index = get_subelement_index(state.subelement)
                subelement_type = state.subelement.split(')')[1].strip()

                value_str = f': {subelement_index}, # ({subelement_type}), evidence: "{evidence.replace('"', '\\"')}"'
                mapping[f'ad_{element}'] = value_str

            for state in sample.maladaptive_states:
                element = state.element.value
                evidence = state.evidence
                subelement_index = get_subelement_index(state.subelement)
                subelement_type = state.subelement.split(')')[1].strip()

                value_str = f': {subelement_index}, # ({subelement_type}), evidence: "{evidence.replace('"', '\\"')}"'
                mapping[f'ma_{element}'] = value_str

            mapping['ad_rating'] = sample.adaptive_presence
            mapping['ma_rating'] = sample.maladaptive_presence

            in_context_str += POST_EXAMPLES.substitute(**mapping) + '\n\n'
        system_prompt = FEW_SHOT_POST_EXAMPLES.substitute(post_examples=in_context_str)
        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'Post: {post.content}'}
        ]

def pick_random_post(dataset: list[Timeline]) -> Post:
    timeline = random.choice(dataset)
    post = random.choice(timeline.posts)
    return post

def get_random_state_vector() -> list[int]:
    state_vector = []
    # adaptive states
    for class_size in ELEMENT_CLASS_SIZE:
        state_vector.append(random.randint(1, class_size))
    # adapative presence
    if sum(state_vector) == 0:
        state_vector.append(1)
    else:
        state_vector.append(random.randint(2, 5))

    # maladaptive states
    for class_size in ELEMENT_CLASS_SIZE:
        state_vector.append(random.randint(1, class_size))
    # maladaptive presence
    if sum(state_vector[-len(ELEMENT_CLASS_SIZE):]) == 0:
        state_vector.append(1)
    else:
        state_vector.append(random.randint(2, 5))
    return state_vector

def query_llm_with_retry(api_address: str, model: str, message: dict[str, str]) -> Any:
    client = OpenAI(api_key='None', base_url=api_address)
    while True:
        try:
            response = query_llm(client, model, message, False)
            llm_state_vector = parse_response(response)
            return llm_state_vector
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(2)

def parallel_query_llm(api_address: str, model: str, messages: list[dict[str, str]]) -> list[Any]:
    with ProcessPoolExecutor(max_workers=min(20, len(messages))) as pool:
        futures = [pool.submit(query_llm_with_retry, api_address, model, message) for message in messages]
        responses = [future.result() for future in futures]
    return responses

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen3.5-27B')
    parser.add_argument('--api_address', type=str, default='http://127.0.0.1:8000/v1')
    parser.add_argument('--train_dir', type=Path)
    parser.add_argument('--test_dir', type=Path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--result_dir', type=Path)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--sampling_type', type=str, choices=['random', 'embedding'], default='random')
    parser.add_argument('--mode', type=str, default='few_shot')
    parser.add_argument('--example_level', type=str, choices=['post', 'subelement'], default='post')

    args = parser.parse_args()
    if not (args.train_dir or args.test_dir):
        print("Error: --train_dir or --test_dir is required")
        exit(1)
    return args


def add_prediction_row(results: pd.DataFrame, post: Post, llm_state_vector: list[int], is_test: bool) -> pd.DataFrame:
    assert len(llm_state_vector) == 2 * len(elements) + 2
    columns = ['timeline_id', 'post_id']
    columns.extend([f'llm_adaptive_{i}' for i in range(1, len(ELEMENT_CLASS_SIZE) + 1)])
    columns.extend([f'llm_adaptive_presence'])
    columns.extend([f'llm_maladaptive_{i}' for i in range(1, len(ELEMENT_CLASS_SIZE) + 1)])
    columns.extend([f'llm_maladaptive_presence'])
    if not is_test:
        columns.extend([f'true_adaptive_{i}' for i in range(1, len(ELEMENT_CLASS_SIZE) + 1)])
        columns.extend([f'true_adaptive_presence'])
        columns.extend([f'true_maladaptive_{i}' for i in range(1, len(ELEMENT_CLASS_SIZE) + 1)])
        columns.extend([f'true_maladaptive_presence'])
        columns.extend([f'random_adaptive_{i}' for i in range(1, len(ELEMENT_CLASS_SIZE) + 1)])
        columns.extend([f'random_adaptive_presence'])
        columns.extend([f'random_maladaptive_{i}' for i in range(1, len(ELEMENT_CLASS_SIZE) + 1)])
        columns.extend([f'random_maladaptive_presence'])
    row = [post.timeline_id, post.post_id] + llm_state_vector
    if not is_test:
        true_state_vector = post.get_state_vector()
        random_state_vector = get_random_state_vector()
        row += true_state_vector + random_state_vector
    row = pd.DataFrame([row], columns=columns)
    results = pd.concat([results, row])
    return results

def get_subelement_evidences(dataset: list[Timeline]) -> pd.DataFrame:
    rows = []
    for timeline in dataset:
        for post in timeline.posts:
            for state in post.adaptive_states:
                rows.append({'timeline_id': timeline.timeline_id, 'post_id': post.post_id, 'self_state': 'adaptive', 'element': state.element.value, 'subelement': state.subelement, 'evidence': state.evidence})
            for state in post.maladaptive_states:
                rows.append({'timeline_id': timeline.timeline_id, 'post_id': post.post_id, 'self_state': 'maladaptive', 'element': state.element.value, 'subelement': state.subelement, 'evidence': state.evidence})
    return pd.DataFrame(rows)

def sample_subelement_evidences(evidences: pd.DataFrame, k: int) -> dict[str, list[str]]:

    placeholder2subelement = {
        'ad_A_1': "(1) Calm/ laid back",
        'ad_A_2': "(3) Sad, emotional pain, grieving",
        'ad_A_3': "(5) Content, happy, joy, hopeful",
        'ad_A_4': "(7) Vigor / energetic",
        'ad_A_5': "(9) Justifiable anger/ assertive anger, justifiable outrage",
        'ad_A_6': "(11) Proud",
        'ad_A_7': "(13) Feeling loved, belong",
        'ma_A_1': "(2) Anxious/ fearful/ tense",
        'ma_A_2': "(4) Depressed, despair, hopeless",
        'ma_A_3': "(6) Mania",
        'ma_A_4': "(8) Apathic, don't care, blunted",
        'ma_A_5': "(10) Angry (aggression), disgust, contempt",
        'ma_A_6': "(12) Ashamed, guilty",
        'ma_A_7': "(14) Feel lonely",

        'ad_BO_1': "(1) Relating behavior",
        'ad_BO_2': "(3) Autonomous or adaptive control behavior",
        'ma_BO_1': "(2) Fight or flight behavior",
        'ma_BO_2': "(4) Over-controlled or controlling behavior",

        'ad_BS_1': "(1) Self care and improvement",
        'ma_BS_1': "(2) Self harm, neglect and avoidance",

        'ad_CO_1': "(1) Perception of the other as related",
        'ad_CO_2': "(3) Perception of the other as facilitating autonomy needs",
        'ma_CO_1': "(2) Perception of the other as detached or over attached",
        'ma_CO_2': "(4) Perception of the other as blocking autonomy needs",

        'ad_CS_1': "(1) Self-acceptance and compassion",
        'ma_CS_1': "(2) Self criticism",

        'ad_D_1': "(1) Relatedness",
        'ad_D_2': "(3) Autonomy and adaptive control",
        'ad_D_3': "(5) Competence, self esteem, self-care",
        'ma_D_1': "(2) Expectation that relatedness needs will not be met",
        'ma_D_2': "(4) Expectation that autonomy needs will not be met",
        'ma_D_3': "(6) Expectation that competence needs will not be met",
    }

    samples = {}

    for placeholder, subelement in placeholder2subelement.items():
        candidates = evidences[evidences['subelement'] == subelement]
        if candidates.empty:
            samples[placeholder] = []
            continue
        sample_size = min(k, len(candidates))
        samples[placeholder] = candidates.sample(sample_size)['evidence'].values.tolist()
    return samples


def main():
    args = parse_args()

    if args.test_dir:
        print(f"Test mode. Test directory: {args.test_dir}")
    else:
        print(f"Train mode")

    result_dir = args.result_dir
    if not result_dir:
        result_dir = Path('results') / str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    result_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=result_dir / 'logging.log', level=logging.INFO, filemode='w')

    train_dataset, test_dataset = load_timelines(args.train_dir, args.test_dir, test=bool(args.test_dir))
    if args.sampling_type == 'embedding':
        save_embeddings(train_dataset)
        if not args.test_dir:
            save_embeddings(test_dataset)

    model = args.model

    results = pd.DataFrame()

    config = {
        'mode': args.mode,  # 'zero_shot' or 'few_shot'
        'sampling_type': args.sampling_type,  # 'random', 'embedding'
        'example_level': args.example_level,  # 'post', 'subelement'
        'k': args.k,
    }

    all_val_posts = [post for timeline in test_dataset for post in timeline.posts]
    all_messages = [make_prompt(post, train_dataset, config) for post in all_val_posts]
    all_llm_state_vectors = parallel_query_llm(args.api_address, model, all_messages)
    for post, llm_state_vector in zip(all_val_posts, all_llm_state_vectors):
        results = add_prediction_row(results, post, llm_state_vector, is_test=bool(args.test_dir))

    results.to_csv(result_dir / f'results.csv', index=False)
if __name__ == '__main__':
    main()