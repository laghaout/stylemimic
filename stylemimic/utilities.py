#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

from collections import defaultdict
from dotenv import load_dotenv, dotenv_values
import json
from mistralai.models.jobs import TrainingParameters
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import os
import tiktoken
from typing import Union


def fine_tune_MistralAI(
    client,
    training_file: str,
    validation_file: str,
    model: str,  # "open-mistral-7b"
    seed: int,
    suffix: str,
) -> dict:
    return client.jobs.create(
        model=model,
        training_files=[training_file],
        validation_files=[validation_file],
        hyperparameters=TrainingParameters(
            training_steps=50,
            learning_rate=1.0e-4,
        ),
    )


def fine_tune_OpenAI(
    client,
    training_file: str,
    validation_file: str,
    model: str,
    seed: int,
    suffix: str,
) -> dict:
    return client.fine_tuning.jobs.create(
        training_file=training_file,
        validation_file=validation_file,
        model=model,
        seed=seed,
        suffix=suffix,
    )


def upload_to_MistralAI(client, file: str) -> dict:
    with open(file, "rb") as f:
        data = client.files.create(file=(file, f))
    return data


def upload_to_OpenAI(client, file: str, purpose: str = "fine-tune") -> dict:
    return client.files.create(file=open(file, "rb"), purpose=purpose)


def row_to_json(
    row,
    key2col: dict = {
        "system": "beat2prose",
        "user": "beat",
        "assistant": "prose",
    },
):
    messages = [
        {"role": "system", "content": row[key2col["system"]]},
        {"role": "user", "content": row[key2col["user"]]},
        {"role": "assistant", "content": row[key2col["assistant"]]},
    ]
    return {"messages": messages}


def tiktoken_count(
    text: str,
    model: str = "gpt-4o",
) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_openai_response(
    client,
    prompt: str,
    system: str,
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    top_p: float = None,
):
    try:
        completion = client.chat.completions.create(
            model=model,
            # response_format={ "type": "json_object" },
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )

        return completion

    except Exception as e:
        return str(e)


def get_mistralai_response(
    client,
    prompt: str,
    system: str,
    model: str,
    temperature: float = None,
    max_tokens: int = None,
    seed: int = None,
    top_p: float = None,
):
    # TODO: Add the parameters temperature, max_tokens, etc.
    try:
        completion = client.chat(
            model=model,
            messages=[
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            random_seed=seed,
            top_p=top_p,
        )

        return completion

    except Exception as e:
        return str(e)


def get_env_vars(
    # env_vars: Union[tuple, None] = None,
) -> dict:
    load_dotenv()
    env_vars = dotenv_values()
    # if env_vars is None:
    #     env_vars = dotenv_values()
    # if isinstance(env_vars, tuple):
    #     env_vars = {var: os.getenv(var) for var in env_vars}

    if os.environ.get("DOCKERIZED", "No") == "Yes":
        data_dir = [env_vars["DATA_DIR_DOCKER"]]
    else:
        data_dir = [".."] + [env_vars["DATA_DIR_LOCAL"]]

    env_vars["DATA_DIR"] = {
        author: os.path.join(*data_dir, author)
        for author in list_subdirectories(data_dir)
    }

    return env_vars


def list_subdirectories(directory: Union[list, tuple], exclude: list = []):
    directory = os.path.join(*directory)
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and name not in exclude
    ]


def validate_JSONL(
    JSONL_path: str,
    model: str,
    max_tokens: int = 16385,
    TARGET_EPOCHS: int = 3,
    MIN_TARGET_EXAMPLES: int = 100,
    MAX_TARGET_EXAMPLES: int = 25000,
    MIN_DEFAULT_EPOCHS: int = 1,
    MAX_DEFAULT_EPOCHS: int = 25,
):
    print("========== DATA VALIDATION:")

    # Load the dataset
    with open(JSONL_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    # %% Data loading
    print("===== Data loading…")

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    # %% Format validation
    print("===== Format validation…")

    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call", "weight")
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(
                content, str
            ):
                format_errors["missing_content"] += 1

        if not any(
            message.get("role", None) == "assistant" for message in messages
        ):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

    # %% Token counting
    print("===== Token counting…")
    encoding = tiktoken.encoding_for_model(model)

    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(
        messages, tokens_per_message=3, tokens_per_name=1
    ):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(
            f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}"
        )

    # %% Data warnings and token counts
    print("===== Data warnings and token counts…")

    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(
            num_assistant_tokens_from_messages(messages)
        )

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(
        assistant_message_lens, "num_assistant_tokens_per_example"
    )
    n_too_long = sum(l > max_tokens for l in convo_lens)
    print(
        f"\n{n_too_long} examples may be over the {max_tokens} token limit, they will be truncated during fine-tuning"
    )

    # %% Cost estimation
    print("===== Cost estimation…")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = max_tokens

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(
            MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples
        )
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(
            MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples
        )

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
    )
    print(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
    )
