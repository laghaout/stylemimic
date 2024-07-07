#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

from dataclasses import dataclass
from datetime import datetime
import json
import os
from openai import OpenAI
import pandas as pd
from typing import Union
import utilities as util


@dataclass
class OneOffWrangler:
    data_dir: str
    author: str
    system: dict
    user: dict  # Text to be prepended to the user prompt
    files: tuple = None  # None gets all files in the author's directory.
    nrows: int = None  # Number of samples
    shuffle_seed: Union[int, None] = None  # None does not shuffle the dataset
    output_size: int = 500  # Number of words in the prose samples
    temperature: float = 0.2
    max_tokens: int = 250
    seed: int = 42  # Seed for the LLM's repeatability
    model: str = "gpt-4o"  # LLM
    verbose: bool = True
    client = OpenAI()

    def validate(self):
        assert 0 <= self.temperature <= 2
        assert self.output_size > 0
        assert self.nrows > 1

    def __post_init__(self) -> None:
        self.validate()

        if self.verbose is True:
            print("========== ONE-OFF WRANGLE:")

        self.dataset = None
        self.datasets = dict(raw=dict())
        self.acquire()
        self.dataset = self.shuffle(self.dataset, self.shuffle_seed)

        # Limit the number of samples if specified.
        self.dataset = self.dataset[: self.nrows]

    def acquire(self) -> None:
        if self.verbose:
            print("===== Acquiring the data…")

        # If the individual files are not specified, get all the *.txt files
        # present in the author's directory.
        if self.files is None:
            self.files = tuple(
                entry
                for entry in os.listdir(self.data_dir[self.author])
                if os.path.isfile(
                    os.path.join(self.data_dir[self.author], entry)
                )
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
            return dataset.sample(frac=1, random_state=shuffle_seed)
        else:
            return dataset

    @staticmethod
    def truncate(
        x, start: bool = False, end: bool = False, sep: str = "."
    ) -> str:
        """Process a sample of text to remove the problematic start/ends."""
        # Split the text at the periods.
        x = x.split(sep)

        # If the start needs to be removed,
        if start == True:
            # remove the first sentence,
            x = x[1:]
            # and skip the white space in the remaining text .
            x = [x[0][1:]] + x[1:]

        # If the end needs to be removed,
        if end == True:
            # remove the last sentence,
            x = x[:-1]
            # and add a period to the penultimate (now last) sentence.
            x = x[:-1] + [x[-1] + sep]

        # Recompse the text with the periods.
        x = sep.join(x)

        return x

    def parse_book(self, filename, encoding="utf-8"):
        # Open a book and save it all in the raw dataset.
        with open(
            os.path.join(self.data_dir[self.author], filename),
            "r",
            encoding=encoding,
        ) as file:
            data = file.read()
        self.datasets["raw"][filename] = data

        # Split the book into paragraphs that are separated by line returns
        # while ignoring sentences with no more than one word.
        data = [k for k in data.split("\n") if len(k.split(" ")) > 1]

        paragraph = []
        cumulative_text = []  # Cumulative text
        length = 0  # Word count

        # For each paragraph...
        for k in data:
            paragraph.append(k)  # add it
            length += len(k.split(" "))
            # so long as the number of words is less than the `output_size`.
            # Otherwise, split the raw text and start over with a new sample.
            if length > self.output_size:
                cumulative_text.append("\n".join(paragraph))
                length = 0
                paragraph = []

        dataset = pd.DataFrame(
            dict(  # TODO: What is going on here?
                beat2prose=self.system["beat2prose"],
                beat=self.user["beat2prose"],
                prose=cumulative_text,
                prose2beat=self.system["prose2beat"],
            )
        )

        # Flag problematic samples:

        # Does not start with an upper case
        dataset["bad_start"] = (
            dataset["prose"]
            .apply(
                lambda x: x[0] != x[0].upper(),
            )
            .astype("bool")
        )

        # Ends with a page number
        dataset["bad_end"] = (
            dataset["prose"]
            .apply(
                lambda x: x.split(" ")[-1][-1] in list("0123456789"),
            )
            .astype("bool")
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
            ["bad_start", "bad_end", "same", "truncated"], inplace=True, axis=1
        )

        dataset.drop_duplicates("prose", inplace=True)

        # Remove any leading or trailing spaces.
        dataset["prose"] = dataset["prose"].apply(lambda x: x.strip())

        return dataset

    def __call__(self):
        """Prepare the dataset with all the necessary prompts."""

        if self.verbose:
            print("===== Populating the beats from the prose…")

        self.completions = pd.DataFrame(
            {
                "prompt": self.dataset["prose"],
                "completion_object": self.dataset["prose"].apply(
                    lambda x: util.get_openai_response(
                        self.client,
                        x,
                        self.system["prose2beat"],
                        self.model,
                        self.temperature,
                        self.max_tokens,
                        self.seed,
                    )
                ),
            }
        )

        self.dataset["beat"] = self.completions["completion_object"].apply(
            lambda x: x.choices[0].message.content.strip()
        )

        self.dataset["beat"] = self.dataset["beat"].apply(
            lambda x: self.user["beat2prose"] + x
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.dataset.to_csv(
            os.path.join(
                self.data_dir[self.author], f"{timestamp} {self.author}.csv"
            )
        )

        json_data = self.dataset.apply(util.row_to_json, axis=1).tolist()

        # Save the list of dictionaries to a JSONL file
        with open(
            os.path.join(
                self.data_dir[self.author], f"{timestamp} {self.author}.jsonl"
            ),
            "w",
            encoding="utf-8",
        ) as outfile:
            for entry in json_data:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write("\n")

        # Validate the JSONL
        util.validate_JSONL(
            os.path.join(
                self.data_dir[self.author], f"{timestamp} {self.author}.jsonl"
            ),
            self.model,
        )
