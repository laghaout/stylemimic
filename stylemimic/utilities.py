#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd

client = OpenAI()

def get_chatgpt_response(
        prompt, system, model, temperature, max_tokens, seed):

    try:
        completion = client.chat.completions.create(
            model=model,
            # response_format={ "type": "json_object" },
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            messages=[
                {"role": "system", 
                 "content": system},
                {"role": "user", 
                 "content": prompt}
            ],
        )

        return completion

    except Exception as e:
        return str(e)

def get_env_vars(env_vars: tuple) -> dict:
    load_dotenv()  

    env_vars = {var: os.getenv(var) for var in env_vars}

    if os.environ.get("DOCKERIZED", "No") == "Yes":
        data_dir = [env_vars["DATA_DIR_DOCKER"]]
    else:
        data_dir = [env_vars["DATA_DIR_LOCAL"]]
        
    env_vars["DATA_DIR"] = {
        author: os.path.join(*data_dir, author) 
        for author in list_subdirectories(data_dir)
    }
    
    return env_vars



def list_subdirectories(directory, exclude=[]):
    directory = os.path.join(*directory)
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and name not in exclude
    ]


