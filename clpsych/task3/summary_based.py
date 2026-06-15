"""
CLPsych 2026; Task 3.1 Sequence Summary Generation
Using vLLM for inference.
"""

import argparse
import json
import os
import re
from pathlib import Path

from openai import OpenAI
from clpsych.common.llm import query_llm


ABCD_LABELS = {
    'A':  {1:'Calm/laid-back', 3:'Sad/grieving', 5:'Content/hopeful', 7:'Vigor/energetic',
           9:'Justifiable anger', 11:'Proud', 13:'Feel loved',
           2:'Anxious/fearful', 4:'Depressed/hopeless', 6:'Mania',
           8:'Apathic', 10:'Angry/contemptuous', 12:'Ashamed/guilty', 14:'Lonely'},
    'B-O':{1:'Relating behavior', 3:'Autonomous behavior',
           2:'Fight or flight', 4:'Over-controlled behavior'},
    'B-S':{1:'Self-care and improvement', 2:'Self-harm/neglect/avoidance'},
    'C-O':{1:'Other as related', 3:'Other as autonomy-facilitating',
           2:'Other as detached', 4:'Other as blocking autonomy'},
    'C-S':{1:'Self-acceptance/compassion', 2:'Self-criticism'},
    'D':  {1:'Relatedness', 3:'Autonomy/adaptive control', 5:'Competence/self-esteem',
           2:'Unmet relatedness expectation', 4:'Unmet autonomy expectation',
           6:'Unmet competence expectation'},
}


def load_tasks12_folder(folder_path: str) -> list[dict]:
    all_posts = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(folder_path, fname), encoding='utf-8') as f:
            data = json.load(f)
        tid = data['timeline_id']
        for p in data['posts']:
            all_posts.append({
                'timeline_id': tid,
                'post_id':     p['post_id'],
                'post_index':  p['post_index'],
                'post_text':   p['post'],
            })
    return all_posts


def format_state(d: dict) -> str:
    if not d or d.get('Presence', 1) <= 1:
        return 'none'
    parts = []
    for el in ['A', 'B-O', 'B-S', 'C-O', 'C-S', 'D']:
        if el in d:
            n = d[el]['subelement']
            parts.append(f"{el}:({n}) {ABCD_LABELS.get(el, {}).get(n, str(n))}")
    return ', '.join(parts) if parts else 'none'


FRAMEWORK_BLOCK = (
    'FRAMEWORK:\n'
    'A self-state is an identifiable unit characterised by specific combinations of ABCD elements.\n'
    'An Adaptive self-state pertains to aspects conducive to fulfillment of basic needs.\n'
    'A Maladaptive self-state pertains to aspects that hinder fulfillment of basic needs.\n\n'
    '1. Affect (A) -- emotional tone:\n'
    '   Adaptive: calm/laid-back; sad/grieving; content/hopeful; vigor/energetic; '
    'justifiable anger; proud; feeling loved\n'
    '   Maladaptive: anxious/fearful; depressed/hopeless; mania; apathic; '
    'angry/contemptuous; ashamed/guilty; lonely\n\n'
    '2. Behavior toward Others (B-O):\n'
    '   Adaptive: relating behavior; autonomous/adaptive control behavior\n'
    '   Maladaptive: fight-or-flight behavior; over-controlled/controlling behavior\n\n'
    '3. Behavior toward Self (B-S):\n'
    '   Adaptive: self-care and improvement\n'
    '   Maladaptive: self-harm, neglect, and avoidance\n\n'
    '4. Cognition of Others (C-O):\n'
    '   Adaptive: perception of others as related; others as facilitating autonomy\n'
    '   Maladaptive: perception of others as detached/over-attached; others as blocking autonomy\n\n'
    '5. Cognition of Self (C-S):\n'
    '   Adaptive: self-acceptance and compassion\n'
    '   Maladaptive: self-criticism\n\n'
    '6. Desire (D) -- needs, expectations, intentions, fears:\n'
    '   Adaptive: relatedness; autonomy/adaptive control; competence/self-esteem\n'
    '   Maladaptive: expectation that relatedness needs will not be met; '
    'expectation that autonomy needs will not be met; '
    'expectation that competence needs will not be met\n\n'
)

CHANGE_DEFS = (
    'CHANGE EVENT DEFINITIONS:\n'
    '- SWITCH: A sudden, substantial change in well-being between two consecutive posts '
    '(|Wellbeing(t) - Wellbeing(t-1)| >= 2). '
    'The change may reflect either improvement or deterioration.\n'
    '- ESCALATION: A gradual intensification of mood over consecutive posts. '
    'The individual progressively shifts from neutral or mildly valenced toward a more extreme state. '
    'May reflect either improvement or deterioration.\n\n'
)

OUTPUT_REQS = (
    'TASK:\n'
    'Generate a structured summary describing patterns of self-state dynamics and their '
    'progression across the sequence surrounding the change event. The summary must describe '
    'how psychological change processes evolve, and how they culminate in (for a Switch) or '
    'unfold through (for an Escalation) the identified change event. '
    'The direction of change (improvement/deterioration) and type of change event '
    '(Switch/Escalation) must be explicitly stated.\n\n'
    'Your summary must address:\n'
    '1. CENTRAL THEME: The dominant ABCD pattern and change trajectory across the sequence. '
    'Describe how the theme appears before the change and how it develops as the change unfolds.\n'
    '2. ADAPTIVE DYNAMICS: How present the adaptive state is and how it changes. '
    'Describe relational dynamics between ABCD subelements within this state '
    '(co-activation, mutual reinforcement, amplification, exacerbation). '
    'If this state is described, relational dynamics between its subelements MUST be described.\n'
    '3. MALADAPTIVE DYNAMICS: Same for the maladaptive state.\n'
    '4. CROSS-STATE DYNAMICS: How the two states relate and how dominance shifts. '
    'This may include one state dominating, suppressing, or silencing the other, '
    'or both coexisting through reflective dialogue. '
    'If cross-state dynamics are present they MUST be described.\n\n'
    'VOCABULARY -- use these exact terms:\n'
    '- Direction: improvement or deterioration\n'
    '- Dynamics: mutually reinforce, co-activate, amplify, overshadow, '
    'suppresses, exacerbates, in reflective dialogue with\n'
    '- Events: a Switch toward [improvement/deterioration] occurs / '
    'an Escalation toward [improvement/deterioration] begins\n\n'
    'CONSTRAINTS:\n'
    '- Always write ABCD abbreviations in parentheses: (A), (B-S), (B-O), (C-S), (C-O), (D)\n'
    '- Do NOT print numeric presence scores\n'
    '- Do NOT reference post indexes\n'
    '- Maximum 350 words\n'
    '- Write ONE paragraph in plain text -- no JSON, no headers\n'
    '- Think step by step: identify dominant ABCD elements per post, identify the change '
    'event and direction, then write your summary\n'
    '- START your summary with the exact phrase: '
    '"The central psychological theme revolves around"\n'
)

PREDS_INSTRUCTION = (
    'STRUCTURED PREDICTIONS are provided for each post:\n'
    '- Switch/Escalation labels: whether a change event is predicted at that post\n'
    '- Adaptive/Maladaptive Presence: how strongly each state is expressed (1=absent to 5=dominant)\n'
    '- ABCD subelements: which specific elements are active in each state\n'
    'Use these predictions as your scaffold. Reference the subelements listed. '
    'Do not invent ABCD elements not present in the predictions.\n\n'
)

SYSTEM_PROMPT_ZERO_SHOT = (
    'You are a clinical psychologist specialising in psychodynamic self-state analysis.\n'
    'Your task is to write a structured sequence summary grounded in the MIND (ABCD) framework.\n\n'
    + FRAMEWORK_BLOCK + CHANGE_DEFS + OUTPUT_REQS
)

SYSTEM_PROMPT_WITH_PREDS = (
    'You are a clinical psychologist specialising in psychodynamic self-state analysis.\n'
    'Your task is to write a structured sequence summary grounded in the MIND (ABCD) framework.\n\n'
    + FRAMEWORK_BLOCK + CHANGE_DEFS + PREDS_INSTRUCTION + OUTPUT_REQS
)

POST_SUMMARY_SYSTEM = (
    'You are a clinical psychologist specialising in psychodynamic self-state analysis.\n'
    'Summarise the interplay between adaptive and maladaptive self-states in this single post.\n'
    'Identify the dominant self-state and describe how the core ABCD elements '
    '(Affect (A), Behavior-Self (B-S), Behavior-Other (B-O), '
    'Cognition-Self (C-S), Cognition-Other (C-O), Desire (D)) interact.\n'
    'Write 2-3 sentences only. Plain text, no headers.\n'
)


def build_chat_prompt(system_content: str, user_content: str) -> str:
    return system_content + '\n\n' + user_content


def clean_output(text: str, max_words: int = 350) -> str:
    m = re.search(r'"summary"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
    return text


def batch_generate(client, model: str, prompts: list[str]) -> list[str]:
    return [query_llm(client, model, [{'role': 'user', 'content': prompt}]) for prompt in prompts]


def generate_post_summaries_batch(client, model: str, postids: list[str], posts_index: dict) -> list[str]:
    valid = [(pid, posts_index[pid]) for pid in postids if pid in posts_index]
    if not valid:
        return []
    prompts = [
        build_chat_prompt(POST_SUMMARY_SYSTEM, f"Post:\n{p['post_text']}")
        for _, p in valid
    ]
    raw_outputs = batch_generate(client, model, prompts)
    return [
        f"Post {p.get('post_index', '?')}: {raw.strip()}"
        for (_, p), raw in zip(valid, raw_outputs)
    ]


def run_sequential_pipeline(client, system_prompt: str, model: str, postids: list[str], posts_index: dict) -> str:
    post_summaries = generate_post_summaries_batch(client, model, postids, posts_index)
    user_msg = (
        'Below are post-level summaries of the self-state dynamics for each post '
        'in a sequence surrounding a mental health change event.\n\n'
        + '\n\n'.join(post_summaries)
        + '\n\nNow write the sequence summary following the output requirements.'
    )
    prompt = build_chat_prompt(system_prompt, user_msg)
    return query_llm(client, model, [{'role': 'user', 'content': prompt}]).strip()


def parse_args():
    parser = argparse.ArgumentParser(description='CLPsych 2026 Task 3.1; vLLM inference')
    parser.add_argument('--api-address', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='Qwen/Qwen3.5-27B')
    parser.add_argument('--posts_dir', type=Path)
    parser.add_argument('--sequences_file', type=Path)
    parser.add_argument('--output_file', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI(api_key='None', base_url=args.api_address)

    print(f'Model: {args.model}')

    with open(args.sequences_file, encoding='utf-8') as f:
        sequences = json.load(f)

    test_posts = load_tasks12_folder(str(args.posts_dir))
    posts_index = {p['post_id']: p for p in test_posts}

    print(f'Sequences: {len(sequences)}  Posts: {len(test_posts)}')

    results = []
    total = len(sequences)
    print(f'\nRunning sequential pipeline on {total} sequences...\n')

    for i, seq in enumerate(sequences, 1):
        tid = seq['timeline_id']
        sid = seq['sequence_id']
        pids = seq['postids']
        print(f'[{i}/{total}] {sid}', end='  ', flush=True)

        summary = run_sequential_pipeline(client, SYSTEM_PROMPT_ZERO_SHOT, args.model, pids, posts_index)
        print(f'{len(summary.split())} words')

        results.append({'timeline_id': tid, 'sequence_id': sid, 'summary': summary})

    print(f'\nDone. {len(results)}/{total} sequences processed.')

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Saved: {args.output_file}')


if __name__ == '__main__':
    main()
