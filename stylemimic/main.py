# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:38:36 2024

@author: Amine Laghaout
"""

from dotenv import load_dotenv
import os
import pandas as pd
import utilities as util


load_dotenv()

# Inside Docker
if os.environ.get("DOCKERIZED", "No") == "Yes":
    suffix = "REM"
    data_dir = [os.getenv(f"data_dir_{suffix}")]
# Outside Docker
else:
    suffix = "LOC"
    data_dir = [os.getenv(f"data_dir_{suffix}")]

data_dir = {
    k: os.path.join(*data_dir, k) for k in util.list_subdirectories(data_dir)
}
OUTPUT_SIZE = 500
START = 10
END = 3554
author = "Sammy"

FILES = [
    entry
    for entry in os.listdir(data_dir[author])
    if os.path.isfile(os.path.join(data_dir[author], entry))
    and entry.split(".")[-1].lower() == "txt"
]


def parse_book(author):
    with open(
        os.path.join(data_dir[author], FILES[0]), "r", encoding="utf-8"
    ) as file:
        data = file.read()

    # Get rid of empty lines and of the header.
    data = [k for k in data.split("\n") if len(k.split(" ")) > 1][START:][:END]

    A = []
    B = []
    length = 0
    for k in data:
        A.append(k)
        length += len(k.split(" "))
        if length > OUTPUT_SIZE:
            B.append("\n".join(A))
            length = 0
            A = []
    data = pd.DataFrame(dict(text=B))

    return data


data = parse_book(author)
