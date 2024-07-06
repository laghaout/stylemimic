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
)

# %%


def main(stage: str, **kwargs):
    if stage == "one-off wrangle":
        oneoff_wrangler = wra.OneOffWrangler(**kwargs)  # Parse the books
        # oneoff_wrangler()  # Generate the beats
        return oneoff_wrangler

    elif stage.split(" ")[0] == "train":
        if stage.split(" ")[1] == "OpenAI":
            learner = lea.LearnerOpenAI(model="gpt-3.5-turbo-1106", **kwargs)
            # learner.train(
            #     upload_JSONL="DATA_TRAIN" not in env_vars.keys()
            #     or "DATA_VALIDATION" not in env_vars.keys()
            # )
        elif stage.split(" ")[1] == "MistralAI":
            learner = lea.LearnerMistralAI(model="open-mistral-7b", **kwargs)
            learner.train(
                # True because apparently one cannot point to file IDs on
                # MistralAI.
                upload_JSONL=True
            )
        else:
            raise ValueError(f"Invalid traing parameter `{stage}`")
        return learner
    elif stage.split(" ")[0] == "serve":
        if stage.split(" ")[1] == "OpenAI":
            learner = lea.LearnerOpenAI(model=kwargs["model"])
            assistant = learner.serve(**kwargs)
        elif stage.split(" ")[1] == "MistralAI":
            learner = lea.LearnerMistralAI(model=kwargs["model"])
            assistant = learner.serve(**kwargs)
        else:
            raise ValueError(f"Invalid traing parameter `{stage}`")

        return assistant
    else:
        raise ValueError(f"Invalid stage `{stage}`")


if __name__ == "__main__":
    # output = main("one-off wrangle", **data_params)
    # output = main("train MistralAI", **modelparams)
    output = main(
        "serve OpenAI",
        **dict(
            system=env_vars["SYSTEM_PROMPT"],
            user=env_vars["USER_PROMPT"],
            model=env_vars["OPENAI_MODEL"],
            temperature=0.3,
            max_tokens=1000,
        ),
    )
    print(output)
