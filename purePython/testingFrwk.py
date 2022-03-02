# region Libraries
import os
import sys
from pprint import pprint

import numpy as np
import random

import v4main as algs

proj_path = os.path.join(
    "/", "home", "rjb255", "University", "ChemEng", "ResearchProject"
)

# sys.path.insert(1, proj_path)
# from purePython.modules.shared.custom import split, getPI, Models

# endregion


def main():

    data_location = os.path.join(proj_path, "data", "big", "qsar_data")

    datasets = os.listdir(data_location)
    random.shuffle(datasets)

    split = [int(len(datasets) * 0.7), int(len(datasets) * 0.85)]
    data_train = datasets[: split[0]]
    data_valid = datasets[split[0] : split[1]]
    data_test = datasets[split[1] :]

    print(f"{len(data_train)}, {len(data_valid)}, {len(data_test)}")


if __name__ == "__main__":
    main()
