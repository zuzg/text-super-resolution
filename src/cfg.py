import yaml


def read_config(path: str) -> dict:
    with open(path) as fp:
        cfg = yaml.safe_load(fp)
    return cfg
