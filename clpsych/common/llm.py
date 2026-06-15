import logging
import unicodedata
from openai import OpenAI

def query_llm(client: OpenAI, model: str, messages: list[dict[str, str]], print_output: bool = False) -> str:
    if 'gpt' in model.lower():
        messages[0]['content'] = messages[0]['content'] + '\nReasoning: high'
    prompt_str = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
    logging.info("Prompt: {}".format(prompt_str))
    args = {}
    if 'qwen3.5' in model.lower():
        extra_body = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        args = {
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
            "extra_body": extra_body,
        }
    elif model.lower() == 'qwen/qwen3-next-80b-a3b-instruct':
        args = {
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
        }
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=20000,
        **args,
        stream=True,
    )
    full_response = []
    for chunk in response:
        output = chunk.choices[0].delta.content
        if output:
            if print_output:
                print(output, end="", flush=True)
            full_response.append(output)

    response_str = "".join(full_response).replace('‑', '-').replace(' ', ' ').replace('’', "'")
    replacements = {
        "\u201c": '"', "\u201d": '"',  # curly quotes
        "\u2018": "'", "\u2019": "'",  # curly apostrophes
        "\u2013": "-", "\u2014": "-",  # en/em dash
        "\u2026": "...",               # ellipsis
    }
    for k, v in replacements.items():
        response_str = response_str.replace(k, v)
    response_str = unicodedata.normalize("NFKD", response_str).encode("ascii", errors="ignore").decode("ascii")
    logging.info("Response: {}".format(response_str))
    return response_str