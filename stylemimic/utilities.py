#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:48:58 2017

@author: Amine Laghaout
"""

import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from types import SimpleNamespace


def list_subdirectories(directory, exclude=[]):
    directory = os.path.join(*directory)
    return [
        name
        for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name)) and name not in exclude
    ]


def version_table(print2screen=True):
    """
    This function returns the version numbers of the various pieces of software
    with which this module was tested.

    TODO: Delocalize this hard-coding into a JSON that can be updated with new
          values.

    Notes
    -----
    In order for Hyperopt 0.1 to work, ``networkx`` had to be downgraded by
    running ``pip install networkx==1.11``. This is due to a bug that arises
    with Hyperopt when version 2.0 of ``networkx`` is installed.
    Also include:
        - conda install plotly
    Parameters
    ----------
    print2screen : bool
        Print the version table to screen (``True``) or return it as a
        dictionary (``False``)?
    Returns
    -------
    version_table : dict
        Dictionary containing the version table
    """

    import cpuinfo  # python -m pip install -U py-cpuinfo
    import platform

    # from dcor import __version__ as dco_version
    from matplotlib import __version__ as plt_version
    from numpy import __version__ as np_version
    from pandas import __version__ as pd_version
    from pytest import __version__ as pyt_version
    from sklearn import __version__ as sk_version
    from sys import version_info
    from tensorflow import __version__ as tf_version

    version_table = {
        "CPU": (
            "Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz",
            cpuinfo.get_cpu_info()["brand_raw"],
        ),
        "CUDA": ("8.0.44", None),
        # "dcor": ("0.6", dco_version),
        "GPU": ("NVIDIA GeForce GTX", None),
        "matplotlib": ("3.6.2", plt_version),
        "NumPy": ("1.23.5", np_version),
        "OS": (
            "Linux-5.10.0-14-amd64-x86_64-with-glibc2.31",
            platform.platform(),
        ),
        "pandas": ("1.5.2", pd_version),
        "PyQt5": ("5.6.2", None),
        "pytest": ("7.3.1", pyt_version),
        "Python": ("3.9.16", ".".join(str(v) for v in version_info[0:3])),
        "sklearn": ("1.2.0", sk_version),
        "TensorFlow.": ("2.11.0", tf_version),
    }

    if print2screen:
        # Maximum length of the software names
        pad = max(map(lambda x: len(x), version_table))

        # Print the table.
        print("software".rjust(pad), ": baseline", sep="")
        print("".rjust(pad), "  current", sep="")
        for k in sorted(version_table.keys()):
            print(k.rjust(pad), ": ", version_table[k][0], sep="")
            print("".rjust(pad), "  ", version_table[k][1], sep="")

    return version_table


def load_learner(path="../lesson/learner.pkl"):
    """Make sure to place yourself in the same directory as `learner.py`"""

    import pickle
    from learners.learner import Learner

    learner = Learner(lesson_dir=None)
    learner = pickle.load(open(path, "rb"))

    return learner


class EnvManager:
    """
    Environment manager. This is used to encapsulate into a single object all
    the parameters that may vary from one environment to another, namely

    - all the paths (both relative and absolute) to the various locations that
      could be relevant (e.g., datasets, models, etc.). Cf. ``self.paths``.
    - All the environment variables that are specific to the particular
      container in which the code is run. Cf. ``self.container``.
    - All the cloud parameters (region, buckets, etc.). Cf. ``self.cloud``.
    - Any other relevant parameter specific to the environment (e.g., the
      username ``self.USER``, etc.)
    """

    def __init__(
        self,
        cloud_params: dict = dict(),
        container_params: tuple = tuple(),
        cloud_params_default: str = "cloud_params.json",
    ):
        """
        Parameters
        ----------
        cloud_params : dict, optional
            Any cloud parameters that need to overwrite the default cloud
            parameters stored in ``cloud_params_default``. The default is
            dict(), i.e., an empty dictionary of such new parameters.
        container_params : tuple, optional
            Any environment parameters that need to be retrieved from the
            container. The default is tuple().
        cloud_params_default : str or list, optional
            Path to the JSON that contains all the cloud parameters. The
            default is 'cloud_params.json'.

        Returns
        -------
        None.
        """

        import getpass

        # Determine the cloud parameters.
        self.cloud_params(cloud_params_default, **cloud_params)

        # Determine the environment parameters in the container.
        self.container_params(container_params)

        # Retrieve the username.
        self.USER = getpass.getuser()

    def __call__(self, **kwargs):
        """
        Once the object has already been populated with its cloud and container
        parameters, one can then create the logic that constructs the
        appropriate paths based on the cloud and container locations.

        The logic is encapsulated in ``self.manage_paths()`` for easier object-
        oriented definition in child classes.

        Parameters
        ----------
        **kwargs : dict
            Dictionary of paths to be constructed where the key is the name of
            the path and the value is its default in a non-containerized, local
            environment.

        Returns
        -------
        None. The result is a SimpleNamespace of the paths stored in
        ``self.paths``.

        """

        self.paths = dict(CWD=os.getcwd())
        for k, v in kwargs.items():
            if isinstance(v, list):
                v = os.path.join(*v)
            self.paths[k] = v
        self.paths = SimpleNamespace(**self.paths)

        self.manage_paths()

    def manage_paths(self):
        """
        Logic which updates the paths based on the container and cloud
        parameters.

        Returns
        -------
        None. This updates ``self.paths``.

        """

        # We are inside the GCP.
        if self.containers.INSIDE_GCP in (True, "Yes"):
            self.paths.data_dir = os.path.join(
                *["/", "gcs", self.cloud.BUCKET, "entity", self.paths.dir_name]
            )
            self.paths.lesson_dir = os.path.join(
                *[
                    "/",
                    "gcs",
                    self.cloud.BUCKET,
                    "map",
                    self.paths.dir_name,
                    self.paths.lesson_dir,
                ]
            )

        # We are outside the GCP, but inside the Docker container.
        elif self.containers.INSIDE_DOCKER_CONTAINER in (True, "Yes"):
            self.paths.data_dir = os.path.join(
                *["/", "home", "Data", self.paths.dir_name]
            )

        # We are neither in the GCP nor in a Docker container.
        else:
            self.paths.data_dir = os.path.join(
                *["/", "home", self.USER, "Data", self.paths.dir_name]
            )

    def container_params(self, args: tuple):
        """
        Determine the tuple of relevant environment variables in the container.

        Parameters
        ----------
        args : tuple
            Tuple of environment variables to be retrieved.

        Returns
        -------
        None. This updates the SimpleNamespace ``self.containers`` of container
        environment variables.

        """

        assert isinstance(args, tuple)

        self.containers = SimpleNamespace(
            **{v: os.environ.get(v, False) for v in args}
        )

    def cloud_params(self, cloud_params_default, **kwargs):
        """
        Parameters
        ----------
        cloud_params : str or list
            Pathname to the JSON that stores the default cloud parameters.
        **kwargs : dict
            Dictionary of cloud parameters that could overwrite the defaults.

        Returns
        -------
        None. This creates a SimpleNamespace ``self.cloud`` with all the
        relevant parameters of the cloud.
        """

        from json import load

        if isinstance(cloud_params_default, list):
            cloud_params_default = os.path.join(*cloud_params_default)
        # TODO: Check that the JSON exists.
        if os.path.isfile(cloud_params_default):
            self.cloud = load(open(cloud_params_default, "r"))
            self.cloud.update(**kwargs)
            self.cloud = SimpleNamespace(**self.cloud)

    def summary(self):
        """
        Summary of all the environment parameters, including the adaptive
        pathnames, from the containers and from the cloud.
        """

        print("\n=== Environment manager summary ========\n")
        for k, v in sorted(self.__dict__.items()):
            if isinstance(v, SimpleNamespace):
                print(f"# {k}")
                for ka, va in v.__dict__.items():
                    print(f"- {ka}: {va}")
            else:
                print(f"- {k}: {v}")
            print()
        print("========================================\n")


def set_argv(defaults, argv):
    """
    This utility function is used to overwrite the default list of arguments
    with those that are passed via ``sys.argv``.

    Parameters
    ----------
    defaults: list
        List of default arguments
    argv: list
        List of arguments to replace the defaults

    Return
    ------
    defaults: list
        Updated list of arguments
    """

    # Note that the first argument is skipped since it is merely the name of
    # the calling function.
    defaults[: len(argv) - 1] = argv[1:]

    return defaults


def args_to_attributes(obj, **kwargs):
    """
    Assign the items of the dictionaries ``default_args`` and ``kwargs`` as
    attributes to the object ``obj``.
    Parameters
    ----------
    obj : object
        Object to which attributes are to be assigned
    kwargs : dict
        Dictionary of attributes to overwrite the defaults.
    """

    [obj.__setattr__(k, kwargs[k]) for k in kwargs.keys()]

    return obj


def ds2df(
    dataset,
    target_name=None,
    take=5,
    join=True,
    codec="utf-8",
    index_col=None,
    exclude_cols=None,
):
    """
    Convert a ``tf.data.Dataset`` to a ``pandas.DataFrame``.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset
    target_name : str, optional
        Label name. The default is None.
    take : int, optional
        Number of batches to take. The default is 5.
    join : bool, optional
        Concatenate all the batches into a single DataFrame? The default is
        False.
    codec : str, optional
        Encoding standard to convert byte strings to string. The default is
        'utf-8'.
    index_col : str, optional
        Index column. The default is None.
    exclude_cols = None, list
        List of columns to exclude from the DataFrame.

    Returns
    -------
    df : pandas.DataFrame, list
        Pandas DataFrame (if ``join=True``) or list (if ``join=False``)
        corresponding to the dataset.

    """

    if isinstance(dataset.element_spec, tuple):
        is_tuple = True
    else:
        is_tuple = False

    dfs = []

    for i, example in enumerate(dataset.take(take)):
        if is_tuple:
            features, targets = example
            if isinstance(exclude_cols, list):
                [features.pop(k) for k in exclude_cols]
            df = pd.DataFrame(features)
            df.insert(0, target_name, targets.numpy())
        else:
            if isinstance(exclude_cols, list):
                [example.pop(k) for k in exclude_cols]
            df = pd.DataFrame(example)

        object_cols = df.select_dtypes([object])
        if not object_cols.empty:
            df[object_cols.columns] = (
                object_cols.stack().str.decode(codec).unstack()
            )

        if index_col is not None:
            df.set_index(index_col, inplace=True)

        dfs += [df]

    if join is not False:
        df = pd.concat(dfs, axis=0)
    else:
        df = dfs

    return df


def select_rows(df, specs):
    """
    Select the rows of the data frame ``df`` as per the columns specified by
    the key-value pairs in ``specs``.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame to to be filtered by column values.
    specs: dict
        Dictionary of column values.

    Return
    ------
    df: pandas.DataFrame
        The input data frame where the rows match the column specifications in
        ``specs``.
    """

    for k in specs.keys():
        if (
            isinstance(specs[k], tuple)
            or isinstance(specs[k], list)
            or isinstance(specs[k], set)
        ):
            df = df.loc[df[k] in specs[k]].copy()
        else:
            df = df.loc[df[k] == specs[k]].copy()

    return df


def rw_data(path, data=None, params=None):
    extension = path.split(".")[-1].lower()

    # Read
    if data is None:
        print(f"Reading from `{path}`…")

        path = open(path, "rb")

        if extension in ("yaml", "yml"):
            import yaml

            if params is None:
                params = dict(Loader=yaml.FullLoader)
            data = yaml.load(path, **params)
        elif extension in ("pickle", "pkl"):
            import pickle

            data = pickle.load(path)
        elif extension in ("json"):
            import json

            data = json.load(path)
        elif extension in ("hdf5", "h5", "hdf"):
            pass
        elif extension in ("csv"):
            data = pd.read_csv(path, **params)
        else:
            print("WARNING: No file format specified.")

        path.close()

        print("done.")

        return data

    # Write
    else:
        print(f"Writing to `{path}`…")

        path = open(path, "wb")

        if extension in ("yaml", "yml"):
            import yaml

            yaml.dump(data, path, default_flow_style=False)
        elif extension in ("pickle", "pkl"):
            import pickle

            pickle.dump(data, path)
        elif extension in ("json"):
            import json

            json.dump(data, fp=path)
        elif extension in ("hdf5", "h5", "hdf"):
            pass
        elif extension in ("csv"):
            data = data.to_csv(path, **params)
        else:
            print("WARNING: No file format specified.")
            return False

        path.close()

        print("done.")

        return True


def score_to_label(score, threshold, smaller_label, larger_label):
    """
    Convert a score to either one of two labels based on a threshold.

    Parameters
    ----------
    score : float
        Score
    threshold : float
        Threshold
    smaller_label : str
        Label to return if the score is strictly smaller than the threshold
    larger_label : TYPE
        Label to return if the score is equal or greater than the threshold

    Returns
    -------
    str
        Label
    """
    return smaller_label if score < threshold else larger_label


def dict_json(x, y=None):
    # Save the dictionary ``x`` to a JSON ``y``.
    if isinstance(x, dict):
        if isinstance(y, str):
            y = [y]
        json.dump(x, fp=open(*y, "w"))

    # Extract the dictionary ``y`` from the JSON ``x``.
    else:
        if isinstance(x, str):
            x = [x]

        return json.load(open(x, "rb"))


def fetch_object(path_from_ref_dir, attributes=None, ref_dir=["/", "home"]):
    """
    Fetch an object from its context.

    Parameters
    ----------
    path_from_ref_dir : list
        Path to the object from the reference directory.
    attribute: str, list of str, None
        If a string, return only the attribute of the object which is labeled
        with that string. If it is a list, return all the attributes in that
        list as a dictrionary. The default is None, in which case the whole
        object is returned.
    ref_dir : TYPE, optional
        Path to the reference directory. The default is ['/', 'home'].

    Returns
    -------
    saved_object : object
        Objecty to be returned
    """

    curr_dir = os.getcwd()

    # Temporarily move to home to view the pickled object in its context.
    os.chdir(os.path.join(*ref_dir))

    # Load the pickled object.
    saved_object = rw_data(os.path.join(*path_from_ref_dir))

    # Move back to the current map you're implementing
    os.chdir(curr_dir)

    if attributes is None:
        return saved_object
    elif isinstance(attributes, str):
        return getattr(saved_object, attributes)
    elif isinstance(attributes, list):
        return {
            attribute: getattr(saved_object, attribute)
            for attribute in attributes
        }


def split(data, parts: dict) -> dict:
    """
    TODO: Adapt this so that it works with pd.DataFrame, NumPy, TensorFlow data
    sets, etc. If using pd.DataFrame, use the split function from sklearn.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset
    parts : dict
        Dictionary of the proportions for the different parts to be split
        into.

    Returns
    -------
    dict
        Dictionary containing all the different parts of the dataset.
    """

    # Make sure the proportions of the different parts add up to unity.
    assert sum(parts.values()) == 1

    # For each part…
    prev = 0
    splitdata = parts.copy()
    lendata = len(data)
    for k, v in parts.items():
        # … take a slice proportional to the part.
        curr = prev + round(v * lendata)
        splitdata[k] = data[prev:curr]
        prev = curr

    return splitdata


def assemble_dataframe(batch, label, target_name="label"):
    """
    Parameters
    ----------
    batch: tf.Tensor
        Features tensor
    label: tf.Tensor
        Labels tensor
    target_name: str
        Name of the label (i.e., target)

    Return
    ------
    batch: pandas.DataFrame
        Data frame which assembles the batch with its labels into one

    TODO: The try~except block is supposed to accommodate the fact that the
    elements_spec is different depending on how the data was generated. Find a
    less hacky way to do this.
    """

    # Use this for tf.Tensors
    try:
        batch = pd.DataFrame(batch.numpy())
        label = pd.DataFrame(label.numpy())
        batch = pd.concat([batch, label], axis=1, sort=False)

    # Use this for make_csv_dataset()
    except Exception:
        batch = pd.DataFrame(batch)
        label = pd.DataFrame({target_name: label}, index=range(len(label)))
        batch = pd.concat([batch, label], axis=1, sort=False)

    batch = pd.DataFrame(batch)

    return batch


def check_docker(verbose=True, BASE_DIR=None):
    import getpass

    # TODO: Replace this function by `get_env_vars()`

    if BASE_DIR is None:
        BASE_DIR = ["/", "home"]

    INSIDE_DOCKER_CONTAINER = os.environ.get("INSIDE_DOCKER_CONTAINER", False)
    USER = getpass.getuser()

    if INSIDE_DOCKER_CONTAINER:
        if verbose:
            print(f"We are inside Docker as USER = {USER}.")
        # matplotlib.use('TkAgg')
    else:
        if verbose:
            print(f"We are outside Docker as USER = {USER}.")
        BASE_DIR += [USER]

    return INSIDE_DOCKER_CONTAINER, USER, BASE_DIR


class PackNumericFeatures(object):
    """
    https://www.tensorflow.org/tutorials/load_data/csv#data_preprocessing
    """

    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [
            tf.cast(feat, tf.float32) for feat in numeric_features
        ]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features["numeric"] = numeric_features

        return features, labels


class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], self.output_dim),
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config["output_dim"] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
