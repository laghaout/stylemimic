#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

from dotenv import load_dotenv
import os
import time
import utilities as util
import wrangler as wra

load_dotenv()

# Inside Docker
if os.environ.get("DOCKERIZED", "No") == "Yes":
    suffix = "REM"
    data_dir = [os.getenv(f"DATA_DIR_{suffix}")]
    lesson_dir = ["lesson"]
# Outside Docker
else:
    suffix = "LOC"
    data_dir = [os.getenv(f"DATA_DIR_{suffix}")]
    lesson_dir = ["..", "lesson"]
data_dir = {
    k: os.path.join(*data_dir, k) for k in util.list_subdirectories(data_dir)
}

# %% Learner class


class Learner:
    def __init__(
        self,
        lesson_dir=lesson_dir,
        data_params=dict(
            data_dir=data_dir,
            output_size=500,
            start=15,  # TODO: Change dynamically
            end=3554,  # TODO: Change dynamically
            author="Sammy",
            model="gpt",
            instruction=dict(
                prose2beat="Sammanfatta följande text till cirka 75 till 100 ord. Sammanfattningen beskriver de viktigaste händelserna med neutral ton."
            ),
        ),
        hyperparams=dict(),
        hyperparams_space=None,
        verbose=1,
        report={
            x: dict()
            for x in ["wrangle", "explore", "select", "train", "test", "serve"]
        },
        **kwargs,
    ):
        """
        Generic learner class.

        In the child class, this can also be used for parameter validation.

        Parameters
        ----------
        lesson_dir: list, tuple, str
            Relative path of the directory where the lesson is stored. If None,
            do not create the dictionary.
        default_lesson_dir: list
            Relative path to the default parent directory.
        report: dict
            Default dictionary of reports from the various stages.
        hyperparams: dict
            Dictionary of hyperparameters
        data_params: dict
            Dictionary of the parameters pertaining to the data or environment.
        hyperparams_space: pd.DataFrame, None, optional
            Hyperparameter space used for model selection.
        """

        # Specify the directory where the learner and its metrics are to be
        # saved.
        if lesson_dir is not None:
            if isinstance(lesson_dir, list) or isinstance(lesson_dir, tuple):
                lesson_dir = os.path.join(*lesson_dir)
            if not os.path.exists(lesson_dir):
                os.makedirs(lesson_dir)

        # Convert all the arguments to attributes.
        util.args_to_attributes(
            self,
            lesson_dir=lesson_dir,
            report=report,
            hyperparams=hyperparams,
            data_params=data_params,
            hyperparams_space=hyperparams_space,
            verbose=verbose,
            **kwargs,
        )

        pass  # Validate the parameters in the child class if necessary.

    def wrangle(self, wrangler_class=wra.Wrangler):
        """
        Prepare the data.

        Note: We're passing wra.Wrangler so as to ensure we're getting the
        child's wrangler class and not the parent class'.
        """

        if self.verbose:
            print("\n========== WRANGLE:")

        delta_tau = time.time()

        # Acquire, validate, and shuffle the raw data.
        self.data = wrangler_class(**self.data_params)

        # Wrangle (i.e., engineer features).
        self.report["wrangle"] = self.data()

        self.report["wrangle"]["delta_tau"] = time.time() - delta_tau

    def design(self):
        """Design the model."""

        if self.verbose:
            print("\n========== DESIGN:")

        self.model = None

    def explore(self):
        """Explore the data."""

        if self.verbose:
            print("\n========== EXPLORE:")

        delta_tau = time.time()

        self.report["explore"] = self.data.explore()

        self.report["explore"]["delta_tau"] = time.time() - delta_tau

    def select(self):
        """Select the model."""

        if self.verbose:
            print("\n========== SELECT:")

        if self.hyperparams_space is None:
            print(
                "WARNING: The hyperparameter space is not specified.",
                "Skipping the model selection.",
            )
            self.report["select"] = None

        pass  # Continue in child class.

    def select_report(self):
        """Report on the model selection."""

        if self.verbose:
            print("===== Selection report:")

        pass  # Continue in child class.

    def train(self):
        """Train the model."""

        if self.verbose:
            print("\n========== TRAIN:")

        pass  # Continue in child class.

    def train_report(self):
        """Report on the training."""

        if self.verbose:
            print("===== Train report:")

        pass  # Continue in child class.

    def test(self):
        """Test the model."""

        if self.verbose:
            print("\n========== TEST:")

        self.report["test"]["metrics"] = None

        pass  # Continue in child class.

    def test_report(self):
        """Report on the testing."""

        if self.verbose:
            print("===== Test report:")

        pass  # Continue in child class.

    def serve(self):
        """Serve the model."""

        if self.verbose:
            print("\n========== SERVE:")

        pass  # Continue in child class.

    def serve_report(self):
        """Report on the serving."""

        if self.verbose:
            print("===== Serve report:")

        pass  # Continue in child class.

    def save(
        self,
        lesson_dir: str = None,
        timestamp: bool = False,
        include_data: bool = False,
        delete_before_save: list = None,
    ) -> None:
        import pickle

        if self.verbose:
            print("\n========== SAVE:")

        if lesson_dir is None and hasattr(self, "lesson_dir"):
            lesson_dir = self.lesson_dir

        if isinstance(timestamp, bool) and timestamp is True:
            timestamp = round(time.time())
            # Note: Alternatively, we can use
            # time.strftime("%Y-%m-%d %H:%M:%S")
        elif timestamp is None or timestamp is False:
            timestamp = ""

        # Save the model.
        if hasattr(self, "model"):
            # Do we have a Keras model?
            try:
                self.model.save(os.path.join(lesson_dir, f"model{timestamp}"))
                print("✓ Saved the model.")
            # If not, just try pickling it.
            except BaseException:
                try:
                    pickle.dump(
                        self.model,
                        open(
                            os.path.join(lesson_dir, f"model{timestamp}.pkl"),
                            "wb",
                        ),
                    )
                    print("✓ Saved the model.")
                except BaseException:
                    print("WARNING: Failed to save the model.")

        # Save the learner object.
        try:
            # Delete any attributes which we do not want to save or which
            # cannot be saved without throwing any error.
            if isinstance(delete_before_save, list):
                assert len(delete_before_save) > 0

                delete_before_save = sorted(delete_before_save)
                delete_before_save.reverse()

                for k in delete_before_save:
                    attributes = k.split(".")
                    if len(attributes) > 1 and hasattr(
                        eval("self." + ".".join(attributes[:-1])),
                        attributes[-1],
                    ):
                        delattr(
                            eval("self." + ".".join(attributes[:-1])),
                            attributes[-1],
                        )
                    elif len(attributes) == 1 and hasattr(self, attributes[0]):
                        delattr(self, attributes[0])
                    else:
                        print(
                            "WARNING: There is a problem with",
                            f"the attribute {k}",
                        )

            # Save whatever remains of the learner object.
            pickle.dump(
                self,
                open(
                    os.path.join(lesson_dir, f"learner{timestamp}.pkl"), "wb"
                ),
            )

            if self.verbose:
                print("✓ Saved the learner.")
        except BaseException:
            print("WARNING: Failed to save the learner.")

    def __call__(
        self,
        explore=False,
        select=False,
        train=True,
        test=True,
        serve=False,
        save=True,
        pause=False,
    ):
        """
        Run the various stages of the learning.

        Parameters
        ----------
        explore: bool
            Explore the data?
        select: bool
            Select the model?
        train: bool
            Train the model?
        test: bool
            Test the model?
        serve: bool
            Serve the model?
        pause: bool
            Pause in between runs?
        """

        if self.verbose:
            print(
                "======================================== start",
                f"[{self.__class__.__name__}]",
            )

        self.wrangle()

        if explore:
            self.explore()
            if pause:
                input("Press Enter to continue.")

        self.design()

        if select:
            self.select()
            self.select_report()
            if pause:
                input("Press Enter to continue.")
        if train:
            self.train()
            self.train_report()
            if pause:
                input("Press Enter to continue.")
        if test:
            self.test()
            self.test_report()
            if pause:
                input("Press Enter to continue.")
        if serve:
            self.serve()
            self.serve_report()

        if save:
            self.save()

        if self.verbose:
            print(
                "======================================== end",
                f"[{self.__class__.__name__}]",
            )


# %% Run as script, not as a module.
if __name__ == "__main__":
    learner = Learner()
    learner()
    report = learner.report
    dataset = learner.data.dataset
    dataset_raw = learner.data.datasets["raw"]
