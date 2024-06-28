# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:38:36 2024

@author: Amine Laghaout
"""

import utilities as util
import wrangler as wra

env_vars = util.get_env_vars(("DATA_DIR_DOCKER", "DATA_DIR_LOCAL", "AUTHOR"))

data_params = dict(
    data_dir=env_vars["DATA_DIR"],
    # files=('Part 1 Ghettokungen.txt',),
    nrows=150,
    shuffle_seed=42,
    author=env_vars["AUTHOR"],
    system=dict(
        prose2beat="Du är en skrivassistent som sammanfattar de viktigaste delarna i en text.",
        beat2prose=f"Du är skrivassistent. När du får en kort sammanfattning av en scen eller av en berättelse kan du veckla ut en berättelse på cirka 500 ord i {env_vars['AUTHOR']} skrivstil.",
        beat2vanillaprose="Du är en skrivassistent som kan veckla ut en berättelse eller en scen utifrån en kort sammanfattning."
    )
)


wrangler = wra.OneOffWrangler(**data_params)
# wrangler()
dataset = wrangler.dataset

# for k in ('prose', 'beat'):
#     dataset[f'len {k}'] = dataset[k].apply(lambda x: len(x.split(' ')))

#%%

# results = pd.DataFrame(
#     {
#         "prompt": prompts,
#         "completion_object": prompts.apply(util.get_chatgpt_response),
#     }
# )
# results["response"] = results["completion_object"].apply(
#     lambda x: x.choices[0].message.content.strip())
# print(results)

