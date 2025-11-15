import json
import re


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


def convert_token_func_to_intervals(token_generation_func: dict, output_tokens: int):
    # calculate the intervals
    intervals = []
    last_interval_type = None
    last_interval_index = None
    data_gen_type = "unwatermarked"
    for index in sorted([int(x) for x in token_generation_func.keys()], reverse=False):
        if last_interval_type is not None:
            intervals.append((last_interval_index, index, last_interval_type))
        last_interval_type = token_generation_func[str(index)].__name__.split("_")[0]
        if last_interval_type != "unwatermarked":
            data_gen_type = last_interval_type
        last_interval_index = index
    intervals.append((last_interval_index, output_tokens, last_interval_type))
    return intervals, data_gen_type
