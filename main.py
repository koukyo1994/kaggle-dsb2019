import logging
import gc
import pickle
import sys
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    sys.path.append("./")

    warnings.filterwarnings("ignore")

    from src.utils import (get_preprocess_parser, load_config,
                           configure_logger, timer, feature_existence_checker,
                           save_json, calc_and_plot_cm)
    from src.features import Basic, generate_features, PastAssessment
    from src.validation import get_validation
    from src.models import get_model

    parser = get_preprocess_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logger(args.config, log_dir=args.log_dir, debug=args.debug)

    logging.info(f"config: {args.config}")
    logging.info(f"debug: {args.debug}")

    config["args"] = dict()
    config["args"]["config"] = args.config

    # make output dir
    output_root_dir = Path(config["output_dir"])
    feature_dir = Path(config["dataset"]["feature_dir"])

    config_name: str = args.config.split("/")[-1].replace(".yml", "")
    output_dir = output_root_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"model output dir: {str(output_dir)}")

    config["model_output_dir"] = str(output_dir)

    # ===============================
    # === Data/Feature Loading
    # ===============================
    input_dir = Path(config["dataset"]["dir"])

    if not feature_existence_checker(feature_dir, config["features"]):
        with timer(name="load data", log=True):
            train = pd.read_csv(input_dir / "train.csv")
            test = pd.read_csv(input_dir / "test.csv")
            specs = pd.read_csv(input_dir / "specs.csv")
            sample_submission = pd.read_csv(
                input_dir / "sample_submission.csv")
        generate_features(
            train, test, namespace=globals(), overwrite=args.force, log=True)

        del train, test
        gc.collect()

    with timer("feature laoding", log=True):
        x_train = pd.concat([
            pd.read_feather(feature_dir / (f + "_train.ftr"), nthreads=-1)
            for f in config["features"]
        ],
                            axis=1,
                            sort=False)
        x_test = pd.concat([
            pd.read_feather(feature_dir / (f + "_test.ftr"), nthreads=-1)
            for f in config["features"]
        ],
                           axis=1,
                           sort=False)

    groups = x_train["installation_id"].values
    y_train = x_train["accuracy_group"].values.reshape(-1)
    cols: List[str] = x_train.columns.tolist()
    cols.remove("installation_id")
    cols.remove("accuracy_group")
    x_train, x_test = x_train[cols], x_test[cols]

    assert len(x_train) == len(y_train)
    logging.debug(f"number of features: {len(cols)}")
    logging.debug(f"number of train samples: {len(x_train)}")
    logging.debug(f"numbber of test samples: {len(x_test)}")

    # ===============================
    # === Adversarial Validation
    # ===============================
    logging.info("Adversarial Validation")
    train_adv = x_train.copy()
    test_adv = x_test.copy()

    train_adv["target"] = 0
    test_adv["target"] = 1
    train_test_adv = pd.concat([train_adv, test_adv], axis=0,
                               sort=False).reset_index(drop=True)

    split_params: dict = config["av"]["split_params"]
    train_set, val_set = train_test_split(
        train_test_adv,
        random_state=split_params["random_state"],
        test_size=split_params["test_size"])
    x_train_adv = train_set[cols]
    y_train_adv = train_set["target"]
    x_val_adv = val_set[cols]
    y_val_adv = val_set["target"]

    logging.debug(f"The number of train set: {len(x_train_adv)}")
    logging.debug(f"The number of valid set: {len(x_val_adv)}")

    train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
    valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

    model_params = config["av"]["model_params"]
    train_params = config["av"]["train_params"]
    clf = lgb.train(
        model_params,
        train_lgb,
        valid_sets=[train_lgb, valid_lgb],
        valid_names=["train", "valid"],
        **train_params)

    # Check the feature importance
    feature_imp = pd.DataFrame(
        sorted(zip(clf.feature_importance(importance_type="gain"), cols)),
        columns=["value", "feature"])

    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="value",
        y="feature",
        data=feature_imp.sort_values(by="value", ascending=False).head(50))
    plt.title("LightGBM Features")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_adv.png")

    config["av_result"] = dict()
    config["av_result"]["score"] = clf.best_score
    config["av_result"]["feature_importances"] = \
        feature_imp.set_index("feature").sort_values(
            by="value",
            ascending=False
        ).head(100).to_dict()["value"]

    # ===============================
    # === Train model
    # ===============================
    logging.info("Train model")

    # get folds
    x_train["group"] = groups
    splits = get_validation(x_train, config)
    x_train.drop("group", axis=1, inplace=True)

    model = get_model(config)
    models, oof_preds, test_preds, feature_importance, eval_results = model.cv(
        y_train, x_train, x_test, cols, splits, config, log=True)

    config["eval_results"] = dict()
    for k, v in eval_results.items():
        config["eval_results"][k] = v

    feature_imp = feature_importance.reset_index().rename(columns={
        "index": "feature",
        0: "value"
    })
    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="value",
        y="feature",
        data=feature_imp.sort_values(by="value", ascending=False).head(50))
    plt.title("Model Features")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_model.png")

    # Confusion Matrix
    calc_and_plot_cm(
        y_train, oof_preds, save_path=output_dir / "confusion_matrix.png")

    # ===============================
    # === Save
    # ===============================
    save_path = output_dir / "output.json"
    save_json(config, save_path)
    np.save(output_dir / "oof_preds.npy", oof_preds)
    with open(output_dir / "model.pkl", "wb") as m:
        pickle.dump(models, m)
