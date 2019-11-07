import logging
import sys
import warnings

import pandas as pd

from pathlib import Path

from easydict import EasyDict as edict

if __name__ == "__main__":
    sys.path.append("./")

    warnings.filterwarnings("ignore")

    from src.utils import (get_preprocess_parser, load_config,
                           configure_logger, timer)

    parser = get_preprocess_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logger(args.config, log_dir=args.log_dir, debug=args.debug)

    logging.info(f"config: {args.config}")
    logging.info(f"debug: {args.debug}")

    config.args = edict()
    config.args.config = args.config

    # make output dir
    output_root_dir = Path(config.output_dir)

    config_name: str = args.config.split("/")[-1].replace(".yml", "")
    output_dir = output_root_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"model output dir: {str(output_dir)}")

    config.model_output_dir = str(output_dir)

    # ===============================
    # === Data Loading
    # ===============================
    with timer(name="load data", log=True):
        input_dir = Path(config.dataset.dir)
        train = pd.read_csv(input_dir / "train.csv")
        test = pd.read_csv(input_dir / "test.csv")
        train_labels = pd.read_csv(input_dir / "train_labels.csv")
        specs = pd.read_csv(input_dir / "specs.csv")
        sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
