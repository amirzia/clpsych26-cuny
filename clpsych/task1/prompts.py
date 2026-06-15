from string import Template


ZERO_SHOT='''### Task:
Your task is to identify adaptive and maladaptive self-states from a reddit post. A post \
can contain zero or more adaptive and maladaptive self-states.

### Definitions
Self-states are conceptualized as structured combinations of 6 elements: Affect (A), \
Behavior of the self with the others (B-O), Behavior of the self toward the self (B-S), \
Cognition of the others (C-O), Cognition of the self (C-S), and Desire (D). Each present \
element has exactly one subelement. The subelements are the specific manifestations of the elements in the post. \
Here are the definitions of the elements and their subelements. \
The percentage of subelements across all posts in the training data is provided in parentheses:
- *Affect* (A): Emotional tone or mood.
    - Adaptive subelements:
        1. Calm / laid back (0.60%)
        2. Sad, emotional pain, grieving (5.36%)
        3. Content, happy, joy, hopeful (8.33%)
        4. Vigor / energetic (0.60%)
        5. Justifiable anger/ assertive anger, justifiable outrage (1.19%)
        6. Proud (4.17%)
        7. Feeling loved, belong (0.00%)
    - Maladaptive subelements:
        1. Anxious/ fearful/ tense (16.67%)
        2. Depressed, despair, hopeless (34.52%)
        3. Mania (0.60%)
        4. Apathic, don't care, blunted (0.00%)
        5. Angry (aggression), disgust, contempt (4.76%)
        6. Ashamed, guilty (4.76%)
        7. Feel lonely (4.76%)
- *Behavior of the self with the others* (B-O): The writer's main behavior(s) toward the others.
    - Adaptive subelements:
        1. Relating behavior (48.21%)
        2. Autonomous or adaptive control behavior
    - Maladaptive subelements:
        1. Fight or flight behavior (13.69%)
        2. Over-controlled or controlling behavior
- *Behavior toward the self* (B-S): The writer's main behavior(s) toward the self.
    - Adaptive subelements:
        1. Self care and improvement (34.52%)
    - Maladaptive subelements:
        1. Self harm, neglect and avoidance (27.98%)
- *Cognition of the others* (C-O): The writer's main perceptions of the other.
    - Adaptive subelements:
        1. Perception of the other as related (19.05%)
        2. Perception of the other as facilitating autonomy needs (1.19%)
    - Maladaptive subelements:
        1. Perception of the other as detached or over attached (45.83%)
        2. Perception of the other as blocking autonomy needs (6.55%)
- *Cognition of the self* (C-S): The writer's main self-perceptions.
    - Adaptive subelements:
        1. Self-acceptance and compassion (25.00%)
    - Maladaptive subelements:
        1. Self criticism (57.14%)
- *Desire* (D): The writer's main desire, expectation, need, intention, or fear.
    - Adaptive subelements:
        1. Relatedness (24.40%)
        2. Autonomy and adaptive control (7.14%)
        3. Competence, self esteem, self-care (21.43%)
    - Maladaptive subelements:
        1. Expectation that relatedness needs will not be met (13.10%)
        2. Expectation that autonomy needs will not be met (4.76%)
        3. Expectation that competence needs will not be met (26.19%)

Self-state rating is the degree to which each identified self state is present in the \
post. It is an integer between 1 and 5 with the following definitions:
- 1 (Not present): The self state is not expressed in the post.
- 2 (Somewhat present): The self state is expressed, but plays a subtle, limited role in \
shaping the person's overall experience.
- 3 (Moderately present): The self state is clearly expressed and moderately contributes \
to the person's experience.
- 4 (Much present): The self state strongly influences and shapes the experience described \
in the post.
- 5 (Highly present): The self state strongly shapes and clearly defines the overall experience \
described in the post.

Percentage of subelements across all posts in the training data is provided in parentheses:
- Affect (A):
    - Adaptive subelements: 1 (0.60%), 2 (5.36%), 3 (8.33%), 4 (0.60%), 5 (1.19%), 6 (4.17%), 7 (0.00%)
    - Maladaptive subelements: 1 (16.67%), 2 (34.52%), 3 (0.60%), 4 (0.00%), 5 (4.76%), 6 (4.76%), 7 (4.76%)
- Behavior of the self with the others (B-O):
    - Adaptive subelements: 1 (48.21%), 2 (4.76%)
    - Maladaptive subelements: 1 (13.69%), 2 (0.00%)
- Behavior toward the self (B-S):
    - Adaptive subelements: 1 (34.52%)
    - Maladaptive subelements: 1 (27.98%)
- Cognition of the others (C-O):
    - Adaptive subelements: 1 (19.05%), 2 (1.19%)
    - Maladaptive subelements: 1 (45.83%), 2 (6.55%)
- Cognition of the self (C-S):
    - Adaptive subelements: 1 (25.00%)
    - Maladaptive subelements: 1 (57.14%)
- Desire (D):
    - Adaptive subelements: 1 (24.40%), 2 (7.14%), 3 (21.43%)
    - Maladaptive subelements: 1 (13.10%), 2 (4.76%), 3 (26.19%)


### Output format
You need to output the presence and rating of the adaptive and maladaptive self-states \
as well as the subelements of the self-states. The subelement of non-existent self-states should be 0. \
Write your output in the following JSON format:
```json
{
    "adaptive_states": {
        "A": int (integer between 0 and 7),
        "B-O": int (integer between 0 and 2),
        "B-S": int (integer between 0 and 1),
        "C-O": int (integer between 0 and 2),
        "C-S": int (integer between 0 and 1),
        "D": int (integer between 0 and 3),
        "rating": int (integer between 1 and 5)
    },
    "maladaptive_states": {
        "A": int (integer between 0 and 7),
        "B-O": int (integer between 0 and 2),
        "B-S": int (integer between 0 and 1),
        "C-O": int (integer between 0 and 2),
        "C-S": int (integer between 0 and 1),
        "D": int (integer between 0 and 3),
        "rating": int (integer between 1 and 5)
    }
}
```
'''

FEW_SHOT_SUBELEMENT_EXAMPLES=Template('''### Task:
Your task is to identify adaptive and maladaptive self-states from a reddit post. A post \
can contain zero or more adaptive and maladaptive self-states.

### Definitions
Self-states are conceptualized as structured combinations of 6 elements: Affect (A), \
Behavior of the self with the others (B-O), Behavior of the self toward the self (B-S), \
Cognition of the others (C-O), Cognition of the self (C-S), and Desire (D). Each present \
element has exactly one subelement. The subelements are the specific manifestations of the elements in the post. \
Here are the definitions of the elements and their subelements. \
The percentage of subelements across all posts in the training data is provided in parentheses:
- *Affect* (A): Emotional tone or mood.
    - Adaptive subelements:
        1. Calm / laid back (0.60%). Examples: $ad_A_1.
        2. Sad, emotional pain, grieving (5.36%). Examples: $ad_A_2.
        3. Content, happy, joy, hopeful (8.33%). Examples: $ad_A_3.
        4. Vigor / energetic (0.60%). Examples: $ad_A_4.
        5. Justifiable anger/ assertive anger, justifiable outrage (1.19%). Examples: $ad_A_5.
        6. Proud (4.17%). Examples: $ad_A_6.
        7. Feeling loved, belong (0.00%). Examples: $ad_A_7.
    - Maladaptive subelements:
        1. Anxious/ fearful/ tense (16.67%). Examples: $ma_A_1.
        2. Depressed, despair, hopeless (34.52%). Examples: $ma_A_2.
        3. Mania (0.60%). Examples: $ma_A_3.
        4. Apathic, don't care, blunted (0.00%). Examples: $ma_A_4.
        5. Angry (aggression), disgust, contempt (4.76%). Examples: $ma_A_5.
        6. Ashamed, guilty (4.76%). Examples: $ma_A_6.
        7. Feel lonely (4.76%). Examples: $ma_A_7.
- *Behavior of the self with the others* (B-O): The writer's main behavior(s) toward the others.
    - Adaptive subelements:
        1. Relating behavior (48.21%). Examples: $ad_BO_1.
        2. Autonomous or adaptive control behavior (4.76%). Examples: $ad_BO_2.
    - Maladaptive subelements:
        1. Fight or flight behavior (13.69%). Examples: $ma_BO_1.
        2. Over-controlled or controlling behavior (0.00%). Examples: $ma_BO_2.
- *Behavior toward the self* (B-S): The writer's main behavior(s) toward the self.
    - Adaptive subelements:
        1. Self care and improvement (34.52%). Examples: $ad_BS_1.
    - Maladaptive subelements:
        1. Self harm, neglect and avoidance (27.98%). Examples: $ma_BS_1.
- *Cognition of the others* (C-O): The writer's main perceptions of the other.
    - Adaptive subelements:
        1. Perception of the other as related (19.05%). Examples: $ad_CO_1.
        2. Perception of the other as facilitating autonomy needs (1.19%). Examples: $ad_CO_2.
    - Maladaptive subelements:
        1. Perception of the other as detached or over attached (45.83%). Examples: $ma_CO_1.
        2. Perception of the other as blocking autonomy needs (6.55%). Examples: $ma_CO_2.
- *Cognition of the self* (C-S): The writer's main self-perceptions.
    - Adaptive subelements:
        1. Self-acceptance and compassion (25.00%). Examples: $ad_CS_1.
    - Maladaptive subelements:
        1. Self criticism (57.14%). Examples: $ma_CS_1.
- *Desire* (D): The writer's main desire, expectation, need, intention, or fear.
    - Adaptive subelements:
        1. Relatedness (24.40%). Examples: $ad_D_1.
        2. Autonomy and adaptive control (7.14%). Examples: $ad_D_2.
        3. Competence, self esteem, self-care (21.43%). Examples: $ad_D_3.
    - Maladaptive subelements:
        1. Expectation that relatedness needs will not be met (13.10%). Examples: $ma_D_1.
        2. Expectation that autonomy needs will not be met (4.76%). Examples: $ma_D_2.
        3. Expectation that competence needs will not be met (26.19%). Examples: $ma_D_3.

Self-state rating is the degree to which each identified self state is present in the \
post. It is an integer between 1 and 5 with the following definitions:
- 1 (Not present): The self state is not expressed in the post.
- 2 (Somewhat present): The self state is expressed, but plays a subtle, limited role in \
shaping the person's overall experience.
- 3 (Moderately present): The self state is clearly expressed and moderately contributes \
to the person's experience.
- 4 (Much present): The self state strongly influences and shapes the experience described \
in the post.
- 5 (Highly present): The self state strongly shapes and clearly defines the overall experience \
described in the post.

Percentage of subelements across all posts in the training data is provided in parentheses:
- Affect (A):
    - Adaptive subelements: 1 (0.60%), 2 (5.36%), 3 (8.33%), 4 (0.60%), 5 (1.19%), 6 (4.17%), 7 (0.00%)
    - Maladaptive subelements: 1 (16.67%), 2 (34.52%), 3 (0.60%), 4 (0.00%), 5 (4.76%), 6 (4.76%), 7 (4.76%)
- Behavior of the self with the others (B-O):
    - Adaptive subelements: 1 (48.21%), 2 (4.76%)
    - Maladaptive subelements: 1 (13.69%), 2 (0.00%)
- Behavior toward the self (B-S):
    - Adaptive subelements: 1 (34.52%)
    - Maladaptive subelements: 1 (27.98%)
- Cognition of the others (C-O):
    - Adaptive subelements: 1 (19.05%), 2 (1.19%)
    - Maladaptive subelements: 1 (45.83%), 2 (6.55%)
- Cognition of the self (C-S):
    - Adaptive subelements: 1 (25.00%)
    - Maladaptive subelements: 1 (57.14%)
- Desire (D):
    - Adaptive subelements: 1 (24.40%), 2 (7.14%), 3 (21.43%)
    - Maladaptive subelements: 1 (13.10%), 2 (4.76%), 3 (26.19%)

### Output format
You need to output the presence and rating of the adaptive and maladaptive self-states \
as well as the subelements of the self-states. The subelement of non-existent self-states should be 0. \
Write your output in the following JSON format:
```json
{
    "adaptive_states": {
        "A": int (integer between 0 and 7),
        "B-O": int (integer between 0 and 2),
        "B-S": int (integer between 0 and 1),
        "C-O": int (integer between 0 and 2),
        "C-S": int (integer between 0 and 1),
        "D": int (integer between 0 and 3),
        "rating": int (integer between 1 and 5)
    },
    "maladaptive_states": {
        "A": int (integer between 0 and 7),
        "B-O": int (integer between 0 and 2),
        "B-S": int (integer between 0 and 1),
        "C-O": int (integer between 0 and 2),
        "C-S": int (integer between 0 and 1),
        "D": int (integer between 0 and 3),
        "rating": int (integer between 1 and 5)
    }
}
```
''')

FEW_SHOT_POST_EXAMPLES=Template('''### Task:
Your task is to identify adaptive and maladaptive self-states from a reddit post. A post \
can contain zero or more adaptive and maladaptive self-states.

### Definitions
Self-states are conceptualized as structured combinations of 6 elements: Affect (A), \
Behavior of the self with the others (B-O), Behavior of the self toward the self (B-S), \
Cognition of the others (C-O), Cognition of the self (C-S), and Desire (D). Each present \
element has exactly one subelement. The subelements are the specific manifestations of the elements in the post. \
Here are the definitions of the elements and their subelements. \
The percentage of subelements across all posts in the training data is provided in parentheses:
1. *Affect* (A): Emotional tone or mood.
    - Adaptive subelements:
        1. Calm / laid back (0.60%)
        2. Sad, emotional pain, grieving (5.36%)
        3. Content, happy, joy, hopeful (8.33%)
        4. Vigor / energetic (0.60%)
        5. Justifiable anger/ assertive anger, justifiable outrage (1.19%)
        6. Proud (4.17%)
        7. Feeling loved, belong (0.00%)
    - Maladaptive subelements:
        1. Anxious/ fearful/ tense (16.67%)
        2. Depressed, despair, hopeless (34.52%)
        3. Mania (0.60%)
        4. Apathic, don't care, blunted (0.00%)
        5. Angry (aggression), disgust, contempt (4.76%)
        6. Ashamed, guilty (4.76%)
        7. Feel lonely (4.76%)
2. *Behavior of the self with the others* (B-O): The writer's main behavior(s) toward the others.
    - Adaptive subelements:
        1. Relating behavior (48.21%).
        2. Autonomous or adaptive control behavior (4.76%).
    - Maladaptive subelements:
        1. Fight or flight behavior (13.69%).
        2. Over-controlled or controlling behavior (0.00%).
3. *Behavior toward the self* (B-S): The writer's main behavior(s) toward the self.
    - Adaptive subelements:
        1. Self care and improvement (34.52%).
    - Maladaptive subelements:
        1. Self harm, neglect and avoidance (27.98%).
4. *Cognition of the others* (C-O): The writer's main perceptions of the other.
    - Adaptive subelements:
        1. Perception of the other as related (19.05%).
        2. Perception of the other as facilitating autonomy needs (1.19%).
    - Maladaptive subelements:
        1. Perception of the other as detached or over attached (45.83%).
        2. Perception of the other as blocking autonomy needs (6.55%).
5. *Cognition of the self* (C-S): The writer's main self-perceptions.
    - Adaptive subelements:
        1. Self-acceptance and compassion (25.00%).
    - Maladaptive subelements:
        1. Self criticism (57.14%).
6. *Desire* (D): The writer's main desire, expectation, need, intention, or fear.
    - Adaptive subelements:
        1. Relatedness (24.40%).
        2. Autonomy and adaptive control (7.14%).
        3. Competence, self esteem, self-care (21.43%).
    - Maladaptive subelements:
        1. Expectation that relatedness needs will not be met (13.10%).
        2. Expectation that autonomy needs will not be met (4.76%).
        3. Expectation that competence needs will not be met (26.19%).

Self-state rating is the degree to which each identified self state is present in the \
post. It is an integer between 1 and 5 with the following definitions:
- 1 (Not present): The self state is not expressed in the post.
- 2 (Somewhat present): The self state is expressed, but plays a subtle, limited role in \
shaping the person's overall experience.
- 3 (Moderately present): The self state is clearly expressed and moderately contributes \
to the person's experience.
- 4 (Much present): The self state strongly influences and shapes the experience described \
in the post.
- 5 (Highly present): The self state strongly shapes and clearly defines the overall experience \
described in the post.

Percentage of subelements across all posts in the training data is provided in parentheses:
- Affect (A):
    - Adaptive subelements: 1 (0.60%), 2 (5.36%), 3 (8.33%), 4 (0.60%), 5 (1.19%), 6 (4.17%), 7 (0.00%)
    - Maladaptive subelements: 1 (16.67%), 2 (34.52%), 3 (0.60%), 4 (0.00%), 5 (4.76%), 6 (4.76%), 7 (4.76%)
- Behavior of the self with the others (B-O):
    - Adaptive subelements: 1 (48.21%), 2 (4.76%)
    - Maladaptive subelements: 1 (13.69%), 2 (0.00%)
- Behavior toward the self (B-S):
    - Adaptive subelements: 1 (34.52%)
    - Maladaptive subelements: 1 (27.98%)
- Cognition of the others (C-O):
    - Adaptive subelements: 1 (19.05%), 2 (1.19%)
    - Maladaptive subelements: 1 (45.83%), 2 (6.55%)
- Cognition of the self (C-S):
    - Adaptive subelements: 1 (25.00%)
    - Maladaptive subelements: 1 (57.14%)
- Desire (D):
    - Adaptive subelements: 1 (24.40%), 2 (7.14%), 3 (21.43%)
    - Maladaptive subelements: 1 (13.10%), 2 (4.76%), 3 (26.19%)

### Examples
Here are some examples of posts and their adaptive and maladaptive self-states:

$post_examples

### Output format
You need to output the presence and rating of the adaptive and maladaptive self-states \
as well as the subelements of the self-states. The subelement of non-existent self-states should be 0. \
The output format is as follows:
```json
{
    "adaptive_states": {
        "A": int (integer between 0 and 7),
        "B-O": int (integer between 0 and 2),
        "B-S": int (integer between 0 and 1),
        "C-O": int (integer between 0 and 2),
        "C-S": int (integer between 0 and 1),
        "D": int (integer between 0 and 3),
        "rating": int (integer between 1 and 5)
    },
    "maladaptive_states": {
        "A": int (integer between 0 and 7),
        "B-O": int (integer between 0 and 2),
        "B-S": int (integer between 0 and 1),
        "C-O": int (integer between 0 and 2),
        "C-S": int (integer between 0 and 1),
        "D": int (integer between 0 and 3),
        "rating": int (integer between 1 and 5)
    }
}
```
''')

POST_EXAMPLES=Template('''\
Post: "$post_content"
Self-states:
{
    "adaptive_states": {
        "A": $ad_A
        "B-O": $ad_BO
        "B-S": $ad_BS
        "C-O": $ad_CO
        "C-S": $ad_CS
        "D": $ad_D
        "rating": $ad_rating
    },
    "maladaptive_states": {
        "A": $ma_A
        "B-O": $ma_BO
        "B-S": $ma_BS
        "C-O": $ma_CO
        "C-S": $ma_CS
        "D": $ma_D
        "rating": $ma_rating
    },
}''')
