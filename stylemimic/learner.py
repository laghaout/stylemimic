# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:35:12 2024

@author: amine
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import time
from typing import Union, Dict
import utilities as util


@dataclass
class LearnerOpenAI:
    data_dir: str
    data_train: str  # Local JSONL or OpenAI ID of the trainig data
    data_validation: str  # Local JSONL or OpenAI ID of the validation data
    temperature: float = 0.2
    max_tokens: int = 500
    seed: int = 42  # Seed for the LLM's repeatability
    model: str = "gpt-4o"  # LLM
    verbose: bool = True
    suffix: Union[str, None] = None
    report: Dict[dict, dict] = field(
        default_factory=lambda: dict(
            upload=dict(delta_tau=None, response=None),
            fine_tune=dict(delta_tau=None, response=None),
        )
    )

    def validate(self):
        assert 0 <= self.temperature <= 2
        assert self.max_tokens > 0

    def __post_init__(self) -> None:
        if self.verbose is True:
            print("========== OpenAI FINE-TUNE:")

        self.validate()

    def __call__(self, upload: bool = False) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # If we do not want to upload any JSONL but simply point to the file
        # IDs that were already uploaded, retrieve them from
        # `data_{train, validation}`.
        if upload is False:
            self.report["upload"]["response"] = dict(
                train={"id": self.data_train},
                validation={"id": self.data_validation},
            )
        # If we do want to upload, `data_{train, validation}` are the paths to
        # the JSONL files.
        else:
            if self.verbose:
                print("===== Upload the training and validation JSONLs…")
            self.report["upload"]["delta_tau"] = time.time()
            self.report["upload"]["response"] = dict(
                train=util.upload_to_OpenAI(self.data_train, "fine-tune"),
                validation=util.upload_to_OpenAI(
                    self.data_validation, "fine-tune"
                ),
            )
            self.report["upload"]["delta_tau"] = (
                time.time() - self.report["upload"]["delta_tau"]
            )
            if self.verbose:
                print(self.report["upload"]["response"])
            with open(
                os.path.join(
                    self.data_dir, f"{timestamp} upload_response.json"
                ),
                "w",
            ) as file:
                json.dump(self.report["upload"]["response"], file, indent=4)

        if self.verbose:
            print("===== Fine-tune…")
        self.report["fine_tune"]["delta_tau"] = time.time()
        self.report["fine_tune"]["response"] = util.fine_tune_OpenAI(
            self.report["upload"]["response"]["train"]["id"],
            self.report["upload"]["response"]["validation"]["id"],
            model=self.model,
            seed=self.seed,
        )
        self.report["fine_tune"]["delta_tau"] = (
            time.time() - self.report["fine_tune"]["delta_tau"]
        )
        if self.verbose:
            print(self.report["fine_tune"]["response"])
        with open(
            os.path.join(
                self.data_dir, f"{timestamp} fine_tune_response.json"
            ),
            "w",
        ) as file:
            json.dump(self.report["fine_tune"]["response"], file, indent=4)
