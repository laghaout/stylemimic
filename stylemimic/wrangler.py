# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 21:13:33 2024

@author: amine
"""

from openai import OpenAI
import os
import pandas as pd
from pydantic import BaseModel
import random
import stylemimic.utilities as util
from tqdm import tqdm
from types import SimpleNamespace
from typing import Tuple, Optional


tqdm.pandas()

CLIENT = OpenAI()


class OneoffWrangler(BaseModel):
    author: str
    data_dir: Tuple[str, ...]
    user_prompt: dict
    system_prompt: str
    token_range: Tuple[int, int]
    random_state: int | None = None
    data: object = None
    max_rows: int | None = None
    x: Optional[dict] = dict()  # Extra parameters

    def model_post_init(self, __context: dict[str, any]) -> None:
        self.x = SimpleNamespace(**dict())

        assert self.token_range[0] <= self.token_range[1]

        self.data = util.load_books(self.data_dir)
        self.x.raw_data = self.data

        data = self.data.split("\n")
        # Get rid of page numbers.
        data = pd.DataFrame(data, columns=["prose"])
        data["chars"] = data["prose"].map(len)
        data["words"] = data["prose"].map(lambda x: len(x.split()))
        data["page_number"] = data["prose"].map(self.page_number)
        data = data[data["chars"] > 1]
        data = data[data["page_number"] == False]
        data.drop(["page_number"], axis=1, inplace=True)
        data["token_count"] = data["prose"].map(util.count_tokens)

        # Clump the text samples by tokens lengths of a random variability.
        clumped_data = []
        token_count = random.randint(self.token_range[0], self.token_range[1])
        prose_sample = ""
        for k, v in list(data.iterrows()):
            prose_sample += v["prose"]
            if token_count - v["token_count"] <= 0:
                clumped_data += [prose_sample]
                prose_sample = ""
                token_count = random.randint(
                    self.token_range[0], self.token_range[1]
                )
            else:
                token_count -= v["token_count"]
                prose_sample += "\n"

        data = pd.DataFrame(clumped_data, columns=["prose"])
        data["token_count"] = data["prose"].map(util.count_tokens)

        if isinstance(self.random_state, int):
            data = data.sample(frac=1, random_state=self.random_state)

        self.data = data.iloc[: self.max_rows]
        util.disp(f"{len(self.data)} prose samples were formed.")

    def __call__(self, prompt_id: str):
        user_prompt = self.user_prompt[prompt_id]

        self.data[f"{prompt_id}_prose"] = self.data["prose"].progress_apply(
            lambda x: util.get_openai_response(
                CLIENT,
                f"{user_prompt}{x}",
                self.system_prompt,
            )
        )

        self.data[f"{prompt_id}_tokens"] = self.data[
            f"{prompt_id}_prose"
        ].apply(util.count_tokens)

        self.data.to_csv(
            os.path.join(*self.data_dir + (f"{self.author}.csv",)),
            sep="\t",
            index=False,
        )

    def disp_user_prompts(self):
        """Display the user prompts."""
        for k, v in self.user_prompt.items():
            print(f"==== {k} ====")
            print()
            print(v)
            print()

    def disp_proses(self, iloc: int = 0):
        """Display all the texts at a given ilocation."""

        for k in [j for j in self.data.columns if j.endswith("prose")]:
            print(f"==== {k} ====")
            print()
            print(self.data.iloc[iloc][k])
            print()

    @staticmethod
    def page_number(x):
        """Check whether the string is an integer."""
        try:
            return isinstance(eval(x), int)
        except BaseException:
            return False
