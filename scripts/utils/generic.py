from typing import Dict, List
import json
import re
import warnings
import traceback
from typing import Iterable, Callable
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm

from utils.watermarking_schemes import (
    BaseWatermark,
    UnWatermark,
    GumbelMaxWatermark,
    InverseWatermark,
    RedGreenWatermark,
    PermuteFlipWatermark,
)
from utils.prf_schemes import prf_factory


# Some more utility functions for different types of metric calculations
def read_json(fpath: str):
    with open(fpath, "r") as f:
        data = json.load(f)
        f.close()
    return data


def normalize_name(name: str):
    name = re.sub(r"[^A-Za-z0-9]", "-", name)
    name = re.sub(r"-+", "-", name)
    return name


def convert_token_func_to_intervals(token_generation_func: Dict[str, BaseWatermark], output_tokens: int):
    # calculate the intervals
    intervals = []
    last_interval_type = None
    last_interval_index = None
    data_gen_type = UnWatermark.__name__  # start with unwatermarked
    for index in sorted([int(x) for x in token_generation_func.keys()], reverse=False):
        if last_interval_type is not None:
            intervals.append((last_interval_index, index, last_interval_type))

        token_func_name = getattr(
            token_generation_func[str(index)], "__name__", token_generation_func[str(index)].__class__.__name__
        )
        last_interval_type = token_func_name
        if token_func_name != UnWatermark.__name__:
            data_gen_type = last_interval_type
        last_interval_index = index
    intervals.append((last_interval_index, output_tokens, last_interval_type))  # append the last interval
    return intervals, data_gen_type


def _safe_run_task(task_func, elem):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ("ok", task_func(elem))
    except Exception as e:
        return ("error", elem, repr(e), traceback.format_exc())


def parallelize_task_loop(iterator, task_func: Callable, n_jobs=5, progress_desc="", stop_on_error=False):
    results = []
    errors = []

    with parallel_backend("loky"):
        with Parallel(
            n_jobs=n_jobs,
            prefer="processes",
            return_as="generator",
            max_nbytes=None,  # avoids mmap edge cases
        ) as parallel:
            tasks = (delayed(_safe_run_task)(task_func, elem) for elem in iterator)

            try:
                for out in tqdm(parallel(tasks), total=len(iterator), desc=progress_desc, leave=True):
                    if out[0] == "ok":
                        results.append(out[1])
                    else:
                        errors.append(out)
                        if stop_on_error:
                            break
            finally:
                pass

    if len(errors) > 0:
        raise Exception(errors[0])
    return results


def generate_watermarking_schemes(
    wm_changes_list: List[int], wm_method: str, vocab_size: int, prf_type: str, key: int = 15485863, **kwargs
) -> Dict[str, BaseWatermark]:
    prf_fun = prf_factory(prf_type)
    wm_dict = {}
    wm_dict["0"] = UnWatermark(vocab_size, prf_fun, key=key)
    wm_mapper = {
        "gumbel": GumbelMaxWatermark,
        "inverse": InverseWatermark,
        "redgreen": RedGreenWatermark,
        "pf": PermuteFlipWatermark,
    }
    for wm in wm_changes_list:
        wm_dict[str(wm)] = wm_mapper[wm_method](vocab_size, prf_fun, key=key, **kwargs)
    return wm_dict
