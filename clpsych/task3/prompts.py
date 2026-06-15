
FEW_SHOT='''You are a clinical psychologist specialising in psychodynamic self-state analysis.
Your task is to write a structured sequence summary grounded in the MIND (ABCD) framework.

# Framework
In the MIND framework, a self-state is defined as an identifiable unit characterized by specific \
combinations of Affect, Behavior (towards the self and others), Cognition (towards the self and others),\
and Desire (ABCD). An Adaptive self-state pertains to aspects of ABCD that are conducive to the \
fulfillment of basic desires/needs. A Maladaptive self-state pertains to aspects of ABCD that hinder \
the fulfillment of basic desires/needs. Each of the ABCD elements is operationalized through a set \
of subelements, representing distinct psychological expressions within each element:

1. *Affect* (A): Emotional tone or mood
    - Adaptive subelements: Calm/ laid back; Sad, emotional pain, grieving; Content, happy, joy, hopeful; Vigor / energetic; Justifiable anger/ assertive anger, justifiable outrage; Proud; Feeling loved, belong
    - Maladaptive subelements: Anxious/ fearful/ tense; Depressed, despair, hopeless; Mania; Apathic, don't care, blunted; Angry (aggression), disgust, contempt; Ashamed, guilt; Feel lonely
2. *Behavior of the self with the others* (B-O): The writer's main behavior(s) toward the others
    - Adaptive subelements: Relating behavior; Autonomous or adaptive control behavior
    - Maladaptive subelements: Fight or flight behavior; Over-controlled or controlling behavior
3. *Behavior toward the self* (B-S): The writer's main behavior(s) toward the self
    - Adaptive subelements: Self care and improvement
    - Maladaptive subelements: Self harm, neglect and avoidance
4. *Cognition of the others* (C-O): The writer's main perceptions of the other
    - Adaptive subelements: Perception of the other as related; Perception of the other as facilitating autonomy needs
    - Maladaptive subelements: Perception of the other as detached or over attached; Perception of the other as blocking autonomy needs
5. *Cognition of the self* (C-S): The writer's main self-perceptions
    - Adaptive subelements: Self-acceptance and compassion
    - Maladaptive subelements: Self criticism
6. *Desire* (D): The writer's main desire, expectation, need, intention, or fear.
    - Adaptive subelements: Relatedness; Autonomy and adaptive control; Competence, self esteem, self-care
    - Maladaptive subelements: Expectation that relatedness needs will not be met; Expectation that autonomy needs will not be met; Expectation that competence needs will not be met

Given a chronologically ordered sequence of posts from a single individual (a timeline), there are two possible clinically meaningful moments of change:

- SWITCH:  A switch reflects a substantial and sudden change in well-being between two consecutive posts. The change may reflect either improvement or deterioration.
- ESCALATION:  An escalation refers to a gradual intensification of mood over a sequence of consecutive posts. It occurs when an individual's mood progressively shifts from neutral or mildly valenced, toward a more extreme state. An escalation may reflect either improvement or deterioration.

# Task
Your task is to generate a structured summary describing patterns of self-state dynamics and their progression over time \
within a sequence of posts surrounding a change (Switch or Escalation). The summary must describe how psychological change \
processes evolve across the sequence, and how they culminate in (when it's a Switch), or unfold through \
(when it's an Escalation), the identified change event. The direction of the change (improvement / deterioration) as well as \
the identity of the change event (Switch/Escalation) should be explicitly stated in the summary. Describe the change pattern using \
the MIND framework (ABCD elements).

The summary should include references, only when they are evident in the data, to the following aspects:
1. *Central recurring theme across the posts*: Describe the central dynamic psychological theme and change \
trajectory characterizing the change process across the sequence. Explain how this theme evolves across \
the sequence. The theme should be described across the stages of the change process within the sequence, \
making clear how the theme appears before the change and how it develops as the change unfolds or when it culminates.
2. *Dynamics within the Adaptive and Maladaptive self-states and their presence*: Describe how present each self-state \
is and how its relative presence changes throughout the sequence as part of the change process. Presence refers to how \
strongly each self-state is expressed or dominant at different points in the sequence, whereas dynamics refer to the \
interactions between the ABCD subelements within that self-state. Where present, describe the adaptive and/or the \
maladaptive self-states in terms of ABCD subelements through explicit relational dynamics between them within the \
same self-state. If a self-state is described, relational dynamics between its ABCD subelements MUST also be \
described. Dynamics within a self-state are relational patterns between two or more subelements within the same \
self-state. These dynamics may be directional or reciprocal, such as co-activation, mutual reinforcement, \
exacerbation of one element by another, amplification of one element by another, or other structured interactions.
3. *Relationship between the adaptive and maladaptive self states and their relative presence*: Describe how \
the adaptive and maladaptive self-states relate to one another and how that changes throughout the sequence. \
Describe how the relative presence and dominance of the adaptive and maladaptive self-states shifts across the \
sequence. This may include: one self-state dominating the other, suppressing or silencing the other, or both \
self-states coexisting through reflective dialogue. examine whether dynamics occur between ABCD subelements across \
opposite self-states (suppression/attenuation, reflective dialogue, dominance competition, resilience or other \
structured interactions). If such cross-self-state dynamics are present in the sequence, they MUST be described.

# Output requirement
Each reference to an ABCD element should include its abbreviation in parentheses. Use the following mapping: \
(A) for Affect; (B-S) for Behavior-self; (B-O) for Behavior-other; (C-S) for Cognition-self; (C-O) for \
Cognition-other; (D) for Desire. Keep your summary below 350 words and write it ONE paragraph. \
Don not mention the post numbers in your summary.

Follow the structre of the examples and write your summary in the same format and start with the term \
"The central psychological theme revolves around..."

Think step by step and analyze the posts. Finally, write your summary in the following format:
```json
{
	"summary": "<YOUR SUMMARY>"
}
```
'''

FEW_SHOT_SIMPLE_PROMPT = '''You are a clinical psychologist specializing in psychodynamic self-state analysis. Write a structured sequence summary grounded in the MIND (ABCD) framework.

# Framework
A self-state is characterized by combinations of Affect (A), Behavior toward others (B-O), Behavior toward self (B-S), Cognition of others (C-O), Cognition of self (C-S), and Desire (D). Self-states are either Adaptive (ABCD elements conducive to fulfilling basic needs) or Maladaptive (ABCD elements that hinder need fulfillment).

Subelements:
- A - Adaptive: calm, sad/grieving, content/hopeful, vigor, justifiable anger, proud, feeling loved. Maladaptive: anxious, depressed/hopeless, manic, apathetic, aggression/contempt, ashamed, lonely.
- B-O - Adaptive: relating, autonomous/adaptive control. Maladaptive: fight/flight, over-controlling.
- B-S - Adaptive: self-care/improvement. Maladaptive: self-harm/neglect/avoidance.
- C-O - Adaptive: other as related, other as autonomy-facilitating. Maladaptive: other as detached/over-attached, other as blocking autonomy.
- C-S - Adaptive: self-acceptance/compassion. Maladaptive: self-criticism.
- D - Adaptive: relatedness, autonomy, competence/self-esteem. Maladaptive: expectation that relatedness, autonomy, or competence needs won't be met.

Change events:
- SWITCH: Sudden, substantial change in well-being between two consecutive posts.
- ESCALATION: Gradual intensification of mood across consecutive posts toward a more extreme state.
Both can reflect improvement or deterioration.

# Task
Write a structured summary describing self-state dynamics and their progression across a chronological post sequence surrounding a change event. Cover:
1. Central recurring theme - how it evolves before and through/culminating in the change.
2. Adaptive and maladaptive self-state dynamics - their relative presence and ABCD subelement interactions (co-activation, reinforcement, amplification, etc.) across the sequence.
3. Relationship between self-states - how dominance, suppression, coexistence, or cross-state dynamics shift across the sequence.

Explicitly state the change direction (improvement/deterioration) and type (Switch/Escalation). Include ABCD abbreviations inline. Stay under 350 words, one paragraph, starting with: "The central psychological theme revolves around..."

Output format:
{
  "summary": "<YOUR SUMMARY>"
}'''


AGGREGATE_PROMPT='''You are a clinical psychologist specialising in psychodynamic self-state analysis.
Your task is to write a structured sequence summary grounded in the MIND (ABCD) framework.

# Framework
In the MIND framework, a self-state is defined as an identifiable unit characterized by specific \
combinations of Affect, Behavior (towards the self and others), Cognition (towards the self and others),\
and Desire (ABCD). An Adaptive self-state pertains to aspects of ABCD that are conducive to the \
fulfillment of basic desires/needs. A Maladaptive self-state pertains to aspects of ABCD that hinder \
the fulfillment of basic desires/needs. Each of the ABCD elements is operationalized through a set \
of subelements, representing distinct psychological expressions within each element:

1. *Affect* (A): Emotional tone or mood
    - Adaptive subelements: Calm/ laid back; Sad, emotional pain, grieving; Content, happy, joy, hopeful; Vigor / energetic; Justifiable anger/ assertive anger, justifiable outrage; Proud; Feeling loved, belong
    - Maladaptive subelements: Anxious/ fearful/ tense; Depressed, despair, hopeless; Mania; Apathic, don't care, blunted; Angry (aggression), disgust, contempt; Ashamed, guilt; Feel lonely
2. *Behavior of the self with the others* (B-O): The writer's main behavior(s) toward the others
    - Adaptive subelements: Relating behavior; Autonomous or adaptive control behavior
    - Maladaptive subelements: Fight or flight behavior; Over-controlled or controlling behavior
3. *Behavior toward the self* (B-S): The writer's main behavior(s) toward the self
    - Adaptive subelements: Self care and improvement
    - Maladaptive subelements: Self harm, neglect and avoidance
4. *Cognition of the others* (C-O): The writer's main perceptions of the other
    - Adaptive subelements: Perception of the other as related; Perception of the other as facilitating autonomy needs
    - Maladaptive subelements: Perception of the other as detached or over attached; Perception of the other as blocking autonomy needs
5. *Cognition of the self* (C-S): The writer's main self-perceptions
    - Adaptive subelements: Self-acceptance and compassion
    - Maladaptive subelements: Self criticism
6. *Desire* (D): The writer's main desire, expectation, need, intention, or fear.
    - Adaptive subelements: Relatedness; Autonomy and adaptive control; Competence, self esteem, self-care
    - Maladaptive subelements: Expectation that relatedness needs will not be met; Expectation that autonomy needs will not be met; Expectation that competence needs will not be met

Given a chronologically ordered sequence of posts from a single individual (a timeline), there are two possible clinically meaningful moments of change:

- SWITCH:  A switch reflects a substantial and sudden change in well-being between two consecutive posts. The change may reflect either improvement or deterioration.
- ESCALATION:  An escalation refers to a gradual intensification of mood over a sequence of consecutive posts. It occurs when an individual's mood progressively shifts from neutral or mildly valenced, toward a more extreme state. An escalation may reflect either improvement or deterioration.

# Task
Your task is to generate a structured summary describing patterns of self-state dynamics and their progression over time \
within a sequence of posts surrounding a change (Switch or Escalation). The summary must describe how psychological change \
processes evolve across the sequence, and how they culminate in (when it's a Switch), or unfold through \
(when it's an Escalation), the identified change event. The direction of the change (improvement / deterioration) as well as \
the identity of the change event (Switch/Escalation) should be explicitly stated in the summary. Describe the change pattern using \
the MIND framework (ABCD elements).

The summary should include references, only when they are evident in the data, to the following aspects:
1. *Central recurring theme across the posts*: Describe the central dynamic psychological theme and change \
trajectory characterizing the change process across the sequence. Explain how this theme evolves across \
the sequence. The theme should be described across the stages of the change process within the sequence, \
making clear how the theme appears before the change and how it develops as the change unfolds or when it culminates.
2. *Dynamics within the Adaptive and Maladaptive self-states and their presence*: Describe how present each self-state \
is and how its relative presence changes throughout the sequence as part of the change process. Presence refers to how \
strongly each self-state is expressed or dominant at different points in the sequence, whereas dynamics refer to the \
interactions between the ABCD subelements within that self-state. Where present, describe the adaptive and/or the \
maladaptive self-states in terms of ABCD subelements through explicit relational dynamics between them within the \
same self-state. If a self-state is described, relational dynamics between its ABCD subelements MUST also be \
described. Dynamics within a self-state are relational patterns between two or more subelements within the same \
self-state. These dynamics may be directional or reciprocal, such as co-activation, mutual reinforcement, \
exacerbation of one element by another, amplification of one element by another, or other structured interactions.
3. *Relationship between the adaptive and maladaptive self states and their relative presence*: Describe how \
the adaptive and maladaptive self-states relate to one another and how that changes throughout the sequence. \
Describe how the relative presence and dominance of the adaptive and maladaptive self-states shifts across the \
sequence. This may include: one self-state dominating the other, suppressing or silencing the other, or both \
self-states coexisting through reflective dialogue. examine whether dynamics occur between ABCD subelements across \
opposite self-states (suppression/attenuation, reflective dialogue, dominance competition, resilience or other \
structured interactions). If such cross-self-state dynamics are present in the sequence, they MUST be described.

You are given a few sample summaries that are written by other LLMs. Refine them to make them more accurate and informative.

# Output requirement
Each reference to an ABCD element should include its abbreviation in parentheses. Use the following mapping: \
(A) for Affect; (B-S) for Behavior-self; (B-O) for Behavior-other; (C-S) for Cognition-self; (C-O) for \
Cognition-other; (D) for Desire. Keep your summary below 350 words and write it ONE paragraph. \
Don not mention the post numbers in your summary.

Follow the structre of the examples and write your summary in the same format and start with the term \
"The central psychological theme revolves around..."

Think step by step and analyze the posts. Finally, write your summary in the following format:
```json
{
	"summary": "<YOUR SUMMARY>"
}
```
'''

JUDGE_AGGREGATE_PROMPT='''You are a clinical psychologist specialising in psychodynamic self-state analysis.
Your task is to write a structured sequence summary grounded in the MIND (ABCD) framework.

# Framework
In the MIND framework, a self-state is defined as an identifiable unit characterized by specific \
combinations of Affect, Behavior (towards the self and others), Cognition (towards the self and others),\
and Desire (ABCD). An Adaptive self-state pertains to aspects of ABCD that are conducive to the \
fulfillment of basic desires/needs. A Maladaptive self-state pertains to aspects of ABCD that hinder \
the fulfillment of basic desires/needs. Each of the ABCD elements is operationalized through a set \
of subelements, representing distinct psychological expressions within each element:

1. *Affect* (A): Emotional tone or mood
    - Adaptive subelements: Calm/ laid back; Sad, emotional pain, grieving; Content, happy, joy, hopeful; Vigor / energetic; Justifiable anger/ assertive anger, justifiable outrage; Proud; Feeling loved, belong
    - Maladaptive subelements: Anxious/ fearful/ tense; Depressed, despair, hopeless; Mania; Apathic, don't care, blunted; Angry (aggression), disgust, contempt; Ashamed, guilt; Feel lonely
2. *Behavior of the self with the others* (B-O): The writer's main behavior(s) toward the others
    - Adaptive subelements: Relating behavior; Autonomous or adaptive control behavior
    - Maladaptive subelements: Fight or flight behavior; Over-controlled or controlling behavior
3. *Behavior toward the self* (B-S): The writer's main behavior(s) toward the self
    - Adaptive subelements: Self care and improvement
    - Maladaptive subelements: Self harm, neglect and avoidance
4. *Cognition of the others* (C-O): The writer's main perceptions of the other
    - Adaptive subelements: Perception of the other as related; Perception of the other as facilitating autonomy needs
    - Maladaptive subelements: Perception of the other as detached or over attached; Perception of the other as blocking autonomy needs
5. *Cognition of the self* (C-S): The writer's main self-perceptions
    - Adaptive subelements: Self-acceptance and compassion
    - Maladaptive subelements: Self criticism
6. *Desire* (D): The writer's main desire, expectation, need, intention, or fear.
    - Adaptive subelements: Relatedness; Autonomy and adaptive control; Competence, self esteem, self-care
    - Maladaptive subelements: Expectation that relatedness needs will not be met; Expectation that autonomy needs will not be met; Expectation that competence needs will not be met

Given a chronologically ordered sequence of posts from a single individual (a timeline), there are two possible clinically meaningful moments of change:

- SWITCH:  A switch reflects a substantial and sudden change in well-being between two consecutive posts. The change may reflect either improvement or deterioration.
- ESCALATION:  An escalation refers to a gradual intensification of mood over a sequence of consecutive posts. It occurs when an individual's mood progressively shifts from neutral or mildly valenced, toward a more extreme state. An escalation may reflect either improvement or deterioration.

# Task
Your task is to generate a structured summary describing patterns of self-state dynamics and their progression over time \
within a sequence of posts surrounding a change (Switch or Escalation). The summary must describe how psychological change \
processes evolve across the sequence, and how they culminate in (when it's a Switch), or unfold through \
(when it's an Escalation), the identified change event. The direction of the change (improvement / deterioration) as well as \
the identity of the change event (Switch/Escalation) should be explicitly stated in the summary. Describe the change pattern using \
the MIND framework (ABCD elements).

The summary should include references, only when they are evident in the data, to the following aspects:
1. *Central recurring theme across the posts*: Describe the central dynamic psychological theme and change \
trajectory characterizing the change process across the sequence. Explain how this theme evolves across \
the sequence. The theme should be described across the stages of the change process within the sequence, \
making clear how the theme appears before the change and how it develops as the change unfolds or when it culminates.
2. *Dynamics within the Adaptive and Maladaptive self-states and their presence*: Describe how present each self-state \
is and how its relative presence changes throughout the sequence as part of the change process. Presence refers to how \
strongly each self-state is expressed or dominant at different points in the sequence, whereas dynamics refer to the \
interactions between the ABCD subelements within that self-state. Where present, describe the adaptive and/or the \
maladaptive self-states in terms of ABCD subelements through explicit relational dynamics between them within the \
same self-state. If a self-state is described, relational dynamics between its ABCD subelements MUST also be \
described. Dynamics within a self-state are relational patterns between two or more subelements within the same \
self-state. These dynamics may be directional or reciprocal, such as co-activation, mutual reinforcement, \
exacerbation of one element by another, amplification of one element by another, or other structured interactions.
3. *Relationship between the adaptive and maladaptive self states and their relative presence*: Describe how \
the adaptive and maladaptive self-states relate to one another and how that changes throughout the sequence. \
Describe how the relative presence and dominance of the adaptive and maladaptive self-states shifts across the \
sequence. This may include: one self-state dominating the other, suppressing or silencing the other, or both \
self-states coexisting through reflective dialogue. examine whether dynamics occur between ABCD subelements across \
opposite self-states (suppression/attenuation, reflective dialogue, dominance competition, resilience or other \
structured interactions). If such cross-self-state dynamics are present in the sequence, they MUST be described.

You are given a few sample summaries that are written by other LLMs. Your task is
to evaluate the given summaries and select the best one.

# Output requirement
Each reference to an ABCD element should include its abbreviation in parentheses. Use the following mapping: \
(A) for Affect; (B-S) for Behavior-self; (B-O) for Behavior-other; (C-S) for Cognition-self; (C-O) for \
Cognition-other; (D) for Desire. Keep your summary below 350 words and write it ONE paragraph. \
Don not mention the post numbers in your summary.

Think step by step and analyze the posts. Finally, rewrite the selected summary in the following format:
```json
{
	"summary": "<SELECTED SUMMARY>"
}
```
'''