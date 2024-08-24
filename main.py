# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:23:14 2024

@author: amine
"""

import pandas as pd
from openai import OpenAI
import os
import stylemimic.utilities as util
from types import SimpleNamespace

client = OpenAI()

#%%

envars = vars(util.get_env_variables())
envars = envars | vars(util.get_env_variables('.env.local'))
envars = SimpleNamespace(**envars)
envars.AUTHORS = envars.AUTHORS.split(',')

USER_PROMPT = "Paraphrase the following text in Swedish in OpenAI's own, neutral style. The length of the re-written text should be approximately the same as the original. (Keep the re-write in Swedish. Do not translate.)"
SYSTEM_PROMPT = "You are a writing assistant that helps re-write texts into OpenAI's writing style"

def wrangle(generate_rewrite: bool = False, max_len: int = None):
    # Concatenate all the rows of the CSV into a single monograph.
    data = {a: pd.DataFrame(pd.read_csv(os.path.join(*(envars.DATA_DIR_LOCAL, a, f"{a}.csv"))))
            for a in envars.AUTHORS}
    data = {a: data[a].sort_values("Unnamed: 0")["prose"] for a in data.keys()}
    data = {a: pd.DataFrame(data[a], columns=['prose']) for a in data.keys()}
    data = {a: '\n'.join(data[a]["prose"]) for a in data.keys()}
    
    # Split into sentences that end with a period.
    data = {a: data[a].split('.') for a in data.keys()}
    data = {a: [j.strip()+'. ' for j in data[a]] for a in data.keys()}
    data = {a: [j.replace('. ”', '.”') for j in data[a]] for a in data.keys()}
    data = {a: util.group_text_into_examples(data[a]) for a in data.keys()}
    data = {a: pd.DataFrame(dict(prose=data[a])).iloc[:max_len] 
            for a in data.keys()}
    for a in data.keys():
        data[a]['len_prose'] = data[a]["prose"].apply(util.count_tokens)
    
    if generate_rewrite:    
        for a in data.keys(): 
            # Prepare the prompt
            data[a]['vanilla_prose'] = data[a]['prose'].apply(
                lambda x: f"{USER_PROMPT}\n\nOriginal text: {x}")
            
            data[a]['vanilla_prose'] = data[a]['vanilla_prose'].apply(
                lambda x: util.get_openai_response(
                    client, 
                    x,
                    SYSTEM_PROMPT))
            
            data[a]['len_vanilla_prose'] = data[a]["vanilla_prose"].apply(util.count_tokens)
            
    return data



data = wrangle(True, max_len=5)

for k in data.keys():
    data[k].to_csv(f'pairs - {k}.csv', sep='\t', index=False)