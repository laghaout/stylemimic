# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:35:12 2024

@author: amine
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from mistralai.client import MistralClient
import os
from openai import OpenAI
import time
from typing import Union, Dict
import utilities as util


@dataclass
class Learner:
    model: str  # LLM
    data_dir: str = None
    data_train: str = None  # Local JSONL or OpenAI ID of the trainig data
    data_validation: str = (
        None  # Local JSONL or OpenAI ID of the validation data
    )
    temperature: float = 0.2
    max_tokens: int = 1000
    seed: int = 42  # Seed for the LLM's repeatability
    # top_p=None,
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
        self.validate()


@dataclass
class LearnerOpenAI(Learner):
    client = OpenAI()

    def train(self, upload_JSONL: bool = True) -> None:
        if self.verbose is True:
            print("========== OpenAI FINE-TUNE:")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # If we do not want to upload any JSONL but simply point to the file
        # IDs that were already uploaded, retrieve them from
        # `data_{train, validation}`.
        if upload_JSONL is False:
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
                train=util.upload_to_OpenAI(
                    self.client, self.data_train, "fine-tune"
                ).__dict__,
                validation=util.upload_to_OpenAI(
                    self.client, self.data_validation, "fine-tune"
                ).__dict__,
            )
            self.report["upload"]["delta_tau"] = (
                time.time() - self.report["upload"]["delta_tau"]
            )
            if self.verbose:
                print(self.report["upload"]["response"])
            with open(
                os.path.join(
                    self.data_dir,
                    "OpenAI",
                    f"{timestamp} upload_response.json",
                ),
                "w",
            ) as file:
                json.dump(self.report["upload"]["response"], file, indent=4)

        if self.verbose:
            print("===== Fine-tune…")
        self.report["fine_tune"]["delta_tau"] = time.time()
        self.report["fine_tune"]["response"] = util.fine_tune_OpenAI(
            self.client,
            self.report["upload"]["response"]["train"]["id"],
            self.report["upload"]["response"]["validation"]["id"],
            model=self.model,
            seed=self.seed,
            suffix=self.suffix,
        )
        self.report["fine_tune"]["delta_tau"] = (
            time.time() - self.report["fine_tune"]["delta_tau"]
        )
        if self.verbose:
            print(self.report["fine_tune"]["response"])
        with open(
            os.path.join(
                self.data_dir, "OpenAI", f"{timestamp} fine_tune_response.json"
            ),
            "w",
        ) as file:
            json.dump(
                {
                    k: v
                    for k, v in self.report["fine_tune"][
                        "response"
                    ].__dict__.items()
                    if k
                    in ("id", "created_at", "training_file", "validation_file")
                },
                file,
                indent=4,
            )

    def serve(
        self,
        user: str,
        system: str,
        model: str = None,
        temperature: float = None,
        seed: int = None,
        max_tokens: int = None,
        **kwargs,
    ) -> str:
        print("========== OpenAI SERVE:")

        output = util.get_openai_response(
            client=self.client,
            prompt=user,
            system=system,
            model=self.model if model is None else model,
            temperature=self.temperature
            if temperature is None
            else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            seed=self.seed if seed is None else seed,
            **kwargs,
        )

        return output.choices[0].message.content.strip()


@dataclass
class LearnerMistralAI(Learner):
    client = MistralClient(util.get_env_vars()["MISTRAL_API_KEY"])

    def train(self, upload_JSONL: bool = True) -> None:
        if self.verbose is True:
            print("========== MistralAI FINE-TUNE:")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # If we do not want to upload any JSONL but simply point to the file
        # IDs that were already uploaded, retrieve them from
        # `data_{train, validation}`.
        if upload_JSONL is False:
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
                train=util.upload_to_MistralAI(
                    self.client,
                    self.data_train,
                ).__dict__,
                validation=util.upload_to_MistralAI(
                    self.client,
                    self.data_validation,
                ).__dict__,
            )
            self.report["upload"]["delta_tau"] = (
                time.time() - self.report["upload"]["delta_tau"]
            )
            if self.verbose:
                print(self.report["upload"]["response"])
            with open(
                os.path.join(
                    self.data_dir,
                    "MistralAI",
                    f"{timestamp} upload_response.json",
                ),
                "w",
            ) as file:
                json.dump(self.report["upload"]["response"], file, indent=4)

        if self.verbose:
            print("===== Fine-tune…")
        self.report["fine_tune"]["delta_tau"] = time.time()
        self.report["fine_tune"]["response"] = util.fine_tune_MistralAI(
            self.client,
            self.report["upload"]["response"]["train"]["id"],
            self.report["upload"]["response"]["validation"]["id"],
            model=self.model,
            seed=self.seed,
            suffix=self.suffix,
        )
        self.report["fine_tune"]["delta_tau"] = (
            time.time() - self.report["fine_tune"]["delta_tau"]
        )
        if self.verbose:
            print(self.report["fine_tune"]["response"])
        with open(
            os.path.join(
                self.data_dir,
                "MistralAI",
                f"{timestamp} fine_tune_response.json",
            ),
            "w",
        ) as file:
            json.dump(
                {
                    k: v
                    for k, v in self.report["fine_tune"][
                        "response"
                    ].__dict__.items()
                    if k
                    in ("id", "created_at", "training_file", "validation_file")
                },
                file,
                indent=4,
            )

    def serve(
        self,
        user: str,
        system: str,
        model: str = None,
        temperature: float = None,
        seed: int = None,
        max_tokens: int = None,
        **kwargs,
    ) -> str:
        print("========== MistralAI SERVE:")

        try:
            model = self.client.jobs.retrieve(model)
            print(f"Using the model from job id `{model}`")
        except BaseException:
            pass

        output = util.get_mistralai_response(
            client=self.client,
            prompt=user,
            system=system,
            model=self.model if model is None else model,
            temperature=self.temperature
            if temperature is None
            else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            seed=self.seed if seed is None else seed,
            **kwargs,
        )

        return output
