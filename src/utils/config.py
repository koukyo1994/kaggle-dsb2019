import yaml

from pathlib import Path
from typing import Optional, Union, Dict, Any


def _get_default() -> dict:
    cfg: Dict[str, Any] = dict()

    # dataset
    cfg["dataset"] = dict()
    cfg["dataset"]["dir"] = "../input"
    cfg["dataset"]["feature_dir"] = "../features"
    cfg["dataset"]["params"] = dict()

    # adversarial validation
    cfg["av"] = dict()
    cfg["av"]["params"] = dict()
    cfg["av"]["split_params"] = dict()
    cfg["av"]["model_params"] = dict()
    cfg["av"]["train_params"] = dict()

    # model
    cfg["model"] = dict()
    cfg["model"]["name"] = "lgbm"
    cfg["model"]["model_params"] = dict()
    cfg["model"]["train_params"] = dict()

    # validation
    cfg["val"] = dict()
    cfg["val"]["name"] = "simple_split"
    cfg["val"]["params"] = dict()

    # others
    cfg["output_dir"] = "../output"

    return cfg


def _merge_config(src: Optional[dict], dst: dict):
    if not isinstance(src, dict):
        return

    for k, v in src.items():
        if isinstance(v, dict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(cfg_path: Optional[Union[str, Path]] = None) -> dict:
    if cfg_path is None:
        config = _get_default()
    else:
        with open(cfg_path, "r") as f:
            cfg = dict(yaml.load(f, Loader=yaml.SafeLoader))

        config = _get_default()
        _merge_config(cfg, config)
    return config
