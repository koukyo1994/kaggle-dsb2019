import yaml

from pathlib import Path
from typing import Optional, Union

from easydict import EasyDict as edict


def _get_default() -> edict:
    cfg = edict()

    # dataset
    cfg.datset = edict()
    cfg.dataset.dir = "../input"
    cfg.dataset.feature_dir = "../features"
    cfg.dataset.params = edict()

    # adversarial validation
    cfg.av = edict()
    cfg.av.params = edict()
    cfg.av.split_params = edict()
    cfg.av.model_params = edict()
    cfg.av.train_params = edict()

    # model
    cfg.model = edict()
    cfg.model.name = "lgbm"
    cfg.model.model_params = edict()
    cfg.model.train_params = edict()

    # validation
    cfg.val = edict()
    cfg.val.name = "simple_split"
    cfg.val.params = edict()

    # others
    cfg.output_dir = "../output"

    return cfg


def _merge_config(src: Optional[edict], dst: edict):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(cfg_path: Optional[Union[str, Path]] = None) -> edict:
    if cfg_path is None:
        config = _get_default()
    else:
        with open(cfg_path, "r") as f:
            cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

        config = _get_default()
        _merge_config(cfg, config)
    return config
