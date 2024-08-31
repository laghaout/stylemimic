# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:11:00 2024

@author: amine
"""

from dotenv import dotenv_values
import logging
import os
import random
import sys
import tiktoken
from types import SimpleNamespace


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_books(data_dir: tuple, extension: str = ".txt"):
    data_dir = os.path.join(*data_dir)

    # List all .txt files in the directory in alphabetical order.
    txt_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
    txt_files.sort()

    # Initialize an empty string to store the concatenated content.
    large_string = ""

    # Concatenate the contents of each file.
    for file_name in txt_files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            large_string += file.read()

    disp(
        f"{count_tokens(large_string)} tokens have been loaded from `{data_dir}`."
    )

    return large_string


def assemble_config(author: str = None):
    envars = vars(get_envars())
    envars = envars | vars(get_envars(".env.local"))
    envars = SimpleNamespace(**envars)
    envars.AUTHORS = envars.AUTHORS.split("|")

    if author is None:
        author = envars.AUTHORS[0]
    elif isinstance(author, int):
        author = envars.AUTHORS[author]

    config = dict(
        oneoffwrangler=dict(
            author=author,
            data_dir=("data", author),
            user_prompt=dict(
                vanilla="Paraphrase the following text in Swedish in OpenAI's own, neutral style. The length of the re-written text should be approximately the same as the original. (Keep the re-write in Swedish. Do not translate.)\n\nOriginal text: ",
                more_vanilla="Please paraphrase the following text. Ensure that all the key information and meaning are preserved, but rephrase the content in a way that it sounds original and distinct from the source. Avoid using the same phrases or structure, while still conveying the same ideas clearly and accurately. (Keep the re-write in Swedish. Do not translate.)\n\nOriginal text: ",
                even_more_vanilla="Please paraphrase the following text in a way that significantly alters the wording and sentence structure while preserving the original meaning. Avoid using similar phrases, and strive for a fresh, original expression of the ideas. The result should convey the same message but sound distinctly different from the original. (Keep the re-write in Swedish. Do not translate.)\n\nOriginal text: ",
                summarized="Summarize the following text in Swedish. Make sure to capture all the key aspects. (Keep the re-write in Swedish. Do not translate.)\n\nOriginal text: ",
            ),
            system_prompt="You are a writing assistant that helps re-write texts into OpenAI's writing style.",
            token_range=(100, 1200),
            random_state=None,
            max_rows=5,
        ),
        envars=envars,
    )

    return config


def disp(text=None, log=False):
    if log:
        logging.info(text)
    else:
        print(text)


def count_tokens(text, model: str = "gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)

    # Tokenize the text
    tokens = tokenizer.encode(text)

    # Count the number of tokens
    num_tokens = len(tokens)

    return num_tokens


def get_envars(env_file: str = ".env"):
    envars = dotenv_values(env_file)
    envars = SimpleNamespace(**envars)
    return envars


def get_openai_response(
    client,
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = "gpt-4o",
    # temperature: float,
    # max_tokens: int,
    seed: int = 42,
    top_p: float = None,
):
    try:
        completion = client.chat.completions.create(
            model=model,
            # response_format={ "type": "json_object" },
            # temperature=temperature,
            # max_tokens=max_tokens,
            seed=seed,
            # top_p=top_p,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )

        return completion.choices[0].message.content

    except Exception as e:
        print("Error", str(e))
        return str(e)
