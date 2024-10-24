# Use this module to code a `pickle_object` function. This will be useful to pickle the model (and encoder if need be).
import pickle
from typing import Any

from prefect import task


@task(name="Save pickle")
def save_pickle(path: str, obj: Any):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@task(name="Load pickle")
def load_pickle(path: str):
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj
