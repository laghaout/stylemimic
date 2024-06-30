# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:38:36 2024

@author: Amine Laghaout
"""

import learner as lea
import os
import utilities as util
import wrangler as wra

env_vars = util.get_env_vars()

data_params = dict(
    data_dir=env_vars["DATA_DIR"],
    # files=('Part 1 Ghettokungen.txt',),
    nrows=200,  # 200
    shuffle_seed=42,
    author=env_vars["AUTHOR"],
    system=dict(
        prose2beat="Du är en skrivassistent som sammanfattar de viktigaste delarna i en text.",
        beat2prose=f"Du är skrivassistent. När du får en sammanfattning av en scen eller av en berättelse utvecklar du den sammanfattningen till en fullständig scen eller berättelse på 500 ord. Texten du skriver imiterar {env_vars['AUTHOR']}s skrivstil.",
    ),
    user=dict(
        beat2prose=f"Skriv en berättelse på cirka 500 ord i {env_vars['AUTHOR']}s skrivstil baserat på följande sammanfattning.\n\nSammanfattning: "
    ),
)

modelparams = dict(
    data_dir=env_vars["DATA_DIR"][env_vars["AUTHOR"]],
    data_train=os.path.join(
        env_vars["DATA_DIR"][env_vars["AUTHOR"]],
        f"{env_vars['AUTHOR']} - train.jsonl",
    )
    if "DATA_TRAIN" not in env_vars.keys()
    else env_vars["DATA_TRAIN"],
    data_validation=os.path.join(
        env_vars["DATA_DIR"][env_vars["AUTHOR"]],
        f"{env_vars['AUTHOR']} - validation.jsonl",
    )
    if "DATA_VALIDATION" not in env_vars.keys()
    else env_vars["DATA_VALIDATION"],
    suffix=env_vars["AUTHOR"].replace(" ", "_").lower(),
    model="gpt-3.5-turbo-1106",
)

# %%


def main(stage: str):
    if stage == "one-off wrangle":
        oneoff_wrangler = wra.OneOffWrangler(**data_params)  # Parse the books
        # oneoff_wrangler()  # Generate the beats
        return oneoff_wrangler

    elif stage == "fine-tune OpenAI":
        learner_OpenAI = lea.LearnerOpenAI(**modelparams)
        learner_OpenAI(
            upload_JSONL="DATA_TRAIN" not in env_vars.keys()
            or "DATA_VALIDATION" not in env_vars.keys()
        )
        return learner_OpenAI
    else:
        raise ValueError(f"Invalid stage `{stage}`")


if __name__ == "__main__":
    output = main("fine-tune OpenAI")
