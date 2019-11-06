import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        default="../config/lgbm_0.yml",
        help="Config file path")
    parser.add_argument(
        "--log_dir", default="../log", help="Directory to save log")
    parser.add_argument(
        "--debug", action="store_true", help="Whether to use debug mode")
    return parser


def get_preprocess_parser() -> argparse.ArgumentParser:
    parser = get_parser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing feature files")
    return parser
