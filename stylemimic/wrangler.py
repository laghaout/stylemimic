#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

from dataclasses import dataclass
import os
import pandas as pd
from typing import Union
import utilities as util

@dataclass
class OneOffWrangler:
    data_dir: str
    author: str
    system: dict 
    files: tuple = None  # None gets all files in the author's directory.
    nrows: int = None
    shuffle_seed: Union[int, None] = None  # None does not shuffle the dataset.
    output_size: int = 500
    temperature: float = 0.2
    max_tokens: int = 500
    seed: int = 42  # Seed for the LLM's repeatability
    model: str = "gpt-4o"  # LLM
    verbose: bool = True
    
    def __post_init__(self) -> None:
        if self.verbose is True:
            print("========== ONE-OFF WRANGLE:")

        self.dataset = None
        self.datasets = dict(raw=dict())
        self.acquire()
        self.dataset = self.shuffle(self.dataset, self.shuffle_seed)

        # Limit the number of samples if specified.
        self.dataset = self.dataset[:self.nrows]

    def acquire(self):
        if self.verbose:
            print("===== Acquiring the data…")

        # If the individual files are not specified, get all the *.txt files
        # present in the author's directory.
        if self.files is None:
            self.files = tuple(
                entry
                for entry in os.listdir(self.data_dir[self.author])
                if os.path.isfile(os.path.join(self.data_dir[self.author], entry))
                and entry.split(".")[-1].lower() == "txt"
            )
        
        self.dataset = pd.concat(
            [self.parse_book(file) for file in self.files],
            axis=0,
            ignore_index=True,
        )

    @staticmethod
    def shuffle(dataset, shuffle_seed: Union[int, None]):
        
        if isinstance(shuffle_seed, int):
            return dataset.sample(
                frac=1, random_state=shuffle_seed)
        else:
            return dataset

    @staticmethod
    def truncate(x, start: bool = False, end: bool = False, sep: str ="."):
        
        # Split the text at the periods.
        x = x.split(sep)
        
        # If the start needs to be removed,
        if start is True:
            # remove the first sentence,
            x = x[1:]
            # and skip the white space in the remaining text .
            x = [x[0][1:]] + x[1:]
        
        # If the end needs to be removed,
        if end is True:
            # remove the last sentence,
            x = x[:-1]
            # and add a period to the penultimate (now last) sentence.
            x = x[:-1] + [x[-1] + sep]
        
        # Recompse the text with the periods.
        x = sep.join(x)
        
        return x

    def parse_book(self, filename, encoding="utf-8"):
        
        with open(
            os.path.join(self.data_dir[self.author], filename),
            "r",
            encoding=encoding,
        ) as file:
            data = file.read()

        # Save the book in the raw dataset.
        self.datasets["raw"][filename] = data

        # Get rid of empty lines and of the header.
        data = [k for k in data.split("\n") if len(k.split(" ")) > 1]

        A = []
        cumulative_text = []  # Cumulative text
        length = 0
        for k in data:
            A.append(k)
            length += len(k.split(" "))
            if length > self.output_size:
                cumulative_text.append("\n".join(A))
                length = 0
                A = []
                
        dataset = pd.DataFrame(
            dict(beat2prose=self.system["beat2prose"], 
                 beat=None, prose=cumulative_text,
                 prose2beat=self.system["prose2beat"])
        )

        # Does not start with an upper case
        dataset["bad_start"] = dataset["prose"].apply(
            lambda x: x[0] != x[0].upper()
        )

        # Ends with a page number
        dataset["bad_end"] = dataset["prose"].apply(
            lambda x: x.split(" ")[-1][-1] in list("0123456789")
        )

        # Truncate the bad parts either at the start or at the end.
        dataset["truncated"] = dataset.apply(
            lambda x: self.truncate(x["prose"], x["bad_start"], x["bad_end"]),
            axis=1,
        )

        dataset["same"] = dataset.apply(
            lambda x: x["truncated"] == x["prose"], axis=1
        )

        dataset["prose"] = dataset["truncated"]

        dataset.drop(
            ["same", "bad_start", "bad_end", "truncated"], inplace=True, axis=1
        )

        dataset.drop_duplicates('prose', inplace=True)

        return dataset

    def __call__(self):
        """ Prepare the dataset with all the necessary prompts. """

        if self.verbose:
            print("===== Populating the beats from the prose…")

        self.completions = pd.DataFrame(
            {
                "prompt": self.dataset['prose'],
                "completion_object": self.dataset['prose'].apply(
                    lambda x: util.get_chatgpt_response(
                        x, self.system["prose2beat"], self.model, self.temperature, 
                        self.max_tokens, self.seed)),
            }
        )
        self.dataset['beat'] = self.completions["completion_object"].apply(
            lambda x: x.choices[0].message.content.strip())