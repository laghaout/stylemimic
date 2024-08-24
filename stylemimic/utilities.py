# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:11:00 2024

@author: amine
"""

from dotenv import dotenv_values
import random
import tiktoken
from types import SimpleNamespace

def group_text_into_examples(text: str, length_range: tuple = (300, 1500)):
    new_text = []
    grouped_text = ''
    for t in text:
        grouped_text += t
        randint = random.randint(
                length_range[0], length_range[1])
        if count_tokens(grouped_text) > randint:
            new_text += [grouped_text]
            grouped_text = ''
            
    return new_text

def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    
    # Tokenize the text
    tokens = tokenizer.encode(text)
    
    # Count the number of tokens
    num_tokens = len(tokens)
    
    return num_tokens

def get_env_variables(env_file: str = '.env'):
    envars = dotenv_values(env_file)
    envars = SimpleNamespace(**envars)
    return envars

def get_openai_response(
    client,
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = 'gpt-4o',
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