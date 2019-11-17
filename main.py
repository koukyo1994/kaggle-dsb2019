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

if __name__ == "__main__":
    sys.path.append("./")

    warnings.filterwarnings("ignore")

    from src.utils import (get_preprocess_parser, load_config,
                           configure_logger, timer, feature_existence_checker,
                           save_json, plot_confusion_matrix, seed_everything)
    from src.features import (Basic, generate_features, PastAssessment,
                              PastClip, PastGame, Unified, ModifiedUnified,
                              UnifiedWithInstallationIDStats)
    from src.validation import get_validation, select_features
    from src.models import get_model

    seed_everything(42)

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
            sample_submission = pd.read_csv(
                input_dir / "sample_submission.csv")
        generate_features(
            train,
            test,
            namespace=globals(),
            required=config["features"],
            overwrite=args.force,
            log=True)

        del train, test
        gc.collect()

    with timer("feature laoding", log=True):
        x_train = pd.concat([
            pd.read_feather(feature_dir / (f + "_train.ftr"), nthreads=-1)
            for f in config["features"]
        ],
                            axis=1,
                            sort=False)
        x_valid = pd.concat([
            pd.read_feather(feature_dir / (f + "_valid.ftr"), nthreads=-1)
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
    groups_valid = x_valid["installation_id"].values
    y_train = x_train["accuracy_group"].values.reshape(-1)
    y_valid = x_valid["accuracy_group"].values.reshape(-1)
    cols: List[str] = x_train.columns.tolist()
    cols.remove("installation_id")
    cols.remove("accuracy_group")
    x_train, x_valid, x_test = x_train[cols], x_valid[cols], x_test[cols]

    assert len(x_train) == len(y_train)
    logging.debug(f"number of features: {len(cols)}")
    logging.debug(f"number of train samples: {len(x_train)}")
    logging.debug(f"numbber of test samples: {len(x_test)}")

    # ===============================
    # === Adversarial Validation
    # ===============================
    logging.info("Adversarial Validation")
    train_adv = x_train.copy()
    test_adv = x_valid.copy()

    train_adv["target"] = 0
    test_adv["target"] = 1
    groups_adv = np.concatenate([groups, groups_valid])
    train_test_adv = pd.concat([train_adv, test_adv], axis=0,
                               sort=False).reset_index(drop=True)

    train_test_adv["group"] = groups_adv
    splits = get_validation(train_test_adv, config)
    train_test_adv.drop("group", axis=1, inplace=True)

    aucs = []
    importance = np.zeros(len(cols))
    for trn_idx, val_idx in splits:
        x_train_adv = train_test_adv.loc[trn_idx, cols]
        y_train_adv = train_test_adv.loc[trn_idx, "target"]
        x_val_adv = train_test_adv.loc[val_idx, cols]
        y_val_adv = train_test_adv.loc[val_idx, "target"]

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

        aucs.append(clf.best_score)
        importance += clf.feature_importance(
            importance_type="gain") / len(splits)

    # Check the feature importance
    feature_imp = pd.DataFrame(
        sorted(zip(importance, cols)), columns=["value", "feature"])

    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="value",
        y="feature",
        data=feature_imp.sort_values(by="value", ascending=False).head(50))
    plt.title("LightGBM Features")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_adv.png")

    config["av_result"] = dict()
    config["av_result"]["score"] = dict()
    for i, auc in enumerate(aucs):
        config["av_result"]["score"][f"fold{i}"] = auc

    config["av_result"]["feature_importances"] = \
        feature_imp.set_index("feature").sort_values(
            by="value",
            ascending=False
        ).to_dict()["value"]

    # ===============================
    # === Train model
    # ===============================
    logging.info("Train model")

    # get folds
    x_train["group"] = groups
    splits = get_validation(x_train, config)
    x_train.drop("group", axis=1, inplace=True)

    cols = select_features(cols, feature_imp, config)

    model = get_model(config)
    models, oof_preds, test_preds, valid_preds, feature_importance, eval_results = model.cv(
        y_train,
        x_train[cols],
        x_test[cols],
        y_valid,
        x_valid[cols],
        feature_name=cols,
        folds_ids=splits,
        config=config,
        log=True)

    config["eval_results"] = dict()
    for k, v in eval_results.items():
        config["eval_results"][k] = v
    if "classwise" not in config["model"]["name"]:
        feature_imp = feature_importance.reset_index().rename(
            columns={
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
    else:
        for k, v in feature_importance.items():
            feature_imp = v.reset_index().rename(columns={
                "index": "feature",
                0: "value"
            })
            plt.figure(figsize=(20, 10))
            sns.barplot(
                x="value",
                y="feature",
                data=feature_imp.sort_values(by="value",
                                             ascending=False).head(50))
            plt.title(f"Feature importance: Assessment {k}")
            plt.tight_layout()
            plt.savefig(output_dir / f"feature_importance_assessment_{k}.png")

    # Confusion Matrix
    plot_confusion_matrix(
        y_train,
        oof_preds,
        classes=np.array(["acc_0", "acc_1", "acc_2", "acc_3"]),
        normalize=True,
        save_path=output_dir / "confusion_matrix_oof.png")
    plot_confusion_matrix(
        y_valid,
        valid_preds,
        classes=np.array(["acc_0", "acc_1", "acc_2", "acc_3"]),
        normalize=True,
        save_path=output_dir / "confusion_matrix_valid.png")

    # ===============================
    # === Save
    # ===============================
    save_path = output_dir / "output.json"
    save_json(config, save_path)
    np.save(output_dir / "oof_preds.npy", oof_preds)
    with open(output_dir / "model.pkl", "wb") as m:
        pickle.dump(models, m)
