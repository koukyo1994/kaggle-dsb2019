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
                           save_json, plot_confusion_matrix, seed_everything,
                           delete_duplicated_columns)
    from src.features import (
        Basic, generate_features, PastAssessment, PastClip, PastGame, Unified,
        ModifiedUnified, UnifiedWithInstallationIDStats, RenewedFeatures,
        PastActivity, ImprovedBasic, ImprovedPastAssessment, ImprovedPastGame,
        PastSummary, PastSummary2, PastSummary3, PastSummary4, NakamaV8, Ratio,
        PastSummary3TimeEncoding, Tfidf, Tfidf2)
    from src.validation import (get_validation, select_features,
                                remove_correlated_features,
                                get_assessment_number)
    from src.models import get_model
    from src.evaluation import (OptimizedRounder,
                                truncated_cv_with_adjustment_of_distribution)

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
        with timer(name="load data"):
            if args.dryrun:
                train = pd.read_csv(input_dir / "train.csv", nrows=50000)
                test = pd.read_csv(input_dir / "test.csv", nrows=50000)
            else:
                train = pd.read_csv(input_dir / "train.csv")
                test = pd.read_csv(input_dir / "test.csv")
            sample_submission = pd.read_csv(
                input_dir / "sample_submission.csv")
        with timer(name="generate features"):
            generate_features(
                train,
                test,
                namespace=globals(),
                required=config["features"],
                overwrite=args.force,
                log=True)

        if globals().get("train") is not None:
            del train, test
            gc.collect()

    if args.dryrun:
        exit(0)

    with timer("feature loading"):
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

    x_train = delete_duplicated_columns(x_train)
    x_valid = delete_duplicated_columns(x_valid)
    x_test = delete_duplicated_columns(x_test)

    groups = x_train["installation_id"].values
    groups_valid = x_valid["installation_id"].values

    test_nth_assessment = get_assessment_number(x_valid, x_test)
    threshold = np.percentile(test_nth_assessment, 95)

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
    # === Feature Selection with correlation
    # ===============================
    # with timer("Feature Selection with correlation"):
    #     to_remove = remove_correlated_features(x_train, cols)

    # cols = [col for col in cols if col not in to_remove]
    logging.info('Training with {} features'.format(len(cols)))
    x_train, x_valid, x_test = x_train[cols], x_valid[cols], x_test[cols]

    # ===============================
    # === Feature Selection with importance
    # ===============================
    # get folds
    # x_train["group"] = groups
    # splits = get_validation(x_train, config)
    # x_train.drop("group", axis=1, inplace=True)

    # feature_selection_config = {
    #     "model": {
    #         "name": "lgbm2",
    #         "mode": "regression",
    #         "sampling": {
    #             "name": "none"
    #         },
    #         "model_params": {
    #             "boosting_type": "gbdt",
    #             "objective": "regression",
    #             "metrics": "rmse",
    #             "max_depth": 6,
    #             "num_leaves": 25,
    #             "learning_rate": 0.01,
    #             "subsample": 0.8,
    #             "subsample_freq": 1,
    #             "colsample_bytree": 0.7,
    #             "data_random_seed": 9999,
    #             "seed": 9999,
    #             "bagging_seed": 9999,
    #             "feature_fraction_seed": 9999,
    #             "reg_alpha": 0.1,
    #             "min_split_gain": 0.5,
    #             "reg_lambda": 0.1,
    #             "min_data_in_leaf": 100,
    #             "n_jobs": -1,
    #             "verbose": -1,
    #             "first_metric_only": True
    #         },
    #         "train_params": {
    #             "num_boost_round": 5000,
    #             "early_stopping_rounds": 100,
    #             "verbose_eval": 100
    #         }
    #     },
    #     "post_process": {
    #         "params": {
    #             "reverse": False,
    #             "n_overall": 20,
    #             "n_classwise": 20
    #         }
    #     }
    # }
    # with timer("Feature Selection with importance"):
    #     model = get_model(feature_selection_config)
    #     _, _, _, _, feature_importance, _ = model.cv(
    #         y_train,
    #         x_train[cols],
    #         x_test[cols],
    #         groups,
    #         feature_name=cols,
    #         folds_ids=splits,
    #         threshold=threshold,
    #         config=feature_selection_config,
    #         log=True)

    #     feature_imp = feature_importance.reset_index().rename(
    #         columns={
    #             "index": "feature",
    #             0: "value"
    #         })
    #     cols = select_features(
    #         cols,
    #         feature_imp,
    #         config,
    #         delete_higher_importance=False)
    #     logging.info(f"Train cols: {len(cols)}")
    #     x_train, x_valid, x_test = x_train[cols], x_valid[cols], x_test[cols]

    # ===============================
    # === Adversarial Validation
    # ===============================
    # logging.info("Adversarial Validation")
    # with timer("Adversarial Validation"):
    #     train_adv = x_train.copy()
    #     test_adv = x_valid.copy()

    #     train_adv["target"] = 0
    #     test_adv["target"] = 1
    #     groups_adv = np.concatenate([groups, groups_valid])
    #     train_test_adv = pd.concat([train_adv, test_adv], axis=0,
    #                                sort=False).reset_index(drop=True)

    #     train_test_adv["group"] = groups_adv
    #     splits = get_validation(train_test_adv, config)
    #     train_test_adv.drop("group", axis=1, inplace=True)

    #     aucs = []
    #     importance = np.zeros(len(cols))
    #     for trn_idx, val_idx in splits:
    #         x_train_adv = train_test_adv.loc[trn_idx, cols]
    #         y_train_adv = train_test_adv.loc[trn_idx, "target"]
    #         x_val_adv = train_test_adv.loc[val_idx, cols]
    #         y_val_adv = train_test_adv.loc[val_idx, "target"]

    #         train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
    #         valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

    #         model_params = config["av"]["model_params"]
    #         train_params = config["av"]["train_params"]
    #         clf = lgb.train(
    #             model_params,
    #             train_lgb,
    #             valid_sets=[train_lgb, valid_lgb],
    #             valid_names=["train", "valid"],
    #             **train_params)

    #         aucs.append(clf.best_score)
    #         importance += clf.feature_importance(
    #             importance_type="gain") / len(splits)

    #     # Check the feature importance
    #     feature_imp = pd.DataFrame(
    #         sorted(zip(importance, cols)), columns=["value", "feature"])

    #     plt.figure(figsize=(20, 10))
    #     sns.barplot(
    #         x="value",
    #         y="feature",
    #         data=feature_imp.sort_values(by="value", ascending=False).head(50))
    #     plt.title("LightGBM Features")
    #     plt.tight_layout()
    #     plt.savefig(output_dir / "feature_importance_adv.png")

    #     config["av_result"] = dict()
    #     config["av_result"]["score"] = dict()
    #     for i, auc in enumerate(aucs):
    #         config["av_result"]["score"][f"fold{i}"] = auc

    #     config["av_result"]["feature_importances"] = \
    #         feature_imp.set_index("feature").sort_values(
    #             by="value",
    #             ascending=False
    #         ).to_dict()["value"]

    # ===============================
    # === Train model
    # ===============================
    logging.info("Train model")

    # get folds
    with timer("Train model"):
        x_train["group"] = groups
        splits = get_validation(x_train, config)
        x_train.drop("group", axis=1, inplace=True)

        model = get_model(config)
        models, oof_preds, y_oof, test_preds, \
            eval_results = model.cv(
                y_train,
                x_train[cols],
                x_test[cols],
                groups,
                feature_name=cols,
                categorical_features=["world", "session_title", "title"],
                folds_ids=splits,
                threshold=threshold,
                config=config)

    config["eval_results"] = dict()
    for k, v in eval_results.items():
        config["eval_results"][k] = v
    # if "classwise" not in config["model"]["name"]:
    #     feature_imp = feature_importance.reset_index().rename(
    #         columns={
    #             "index": "feature",
    #             0: "value"
    #         })
    #     plt.figure(figsize=(20, 10))
    #     sns.barplot(
    #         x="value",
    #         y="feature",
    #         data=feature_imp.sort_values(by="value", ascending=False).head(50))
    #     plt.title("Model Features")
    #     plt.tight_layout()
    #     plt.savefig(output_dir / "feature_importance_model.png")
    # else:
    #     for k, v in feature_importance.items():
    #         feature_imp = v.reset_index().rename(columns={
    #             "index": "feature",
    #             0: "value"
    #         })
    #         plt.figure(figsize=(20, 10))
    #         sns.barplot(
    #             x="value",
    #             y="feature",
    #             data=feature_imp.sort_values(by="value",
    #                                          ascending=False).head(50))
    #         plt.title(f"Feature importance: Assessment {k}")
    #         plt.tight_layout()
    #         plt.savefig(output_dir / f"feature_importance_assessment_{k}.png")

    # Confusion Matrix
    plot_confusion_matrix(
        y_oof,
        oof_preds,
        classes=np.array(["acc_0", "acc_1", "acc_2", "acc_3"]),
        normalize=True,
        save_path=output_dir / "confusion_matrix_oof.png")

    raw_normal_oof = model.raw_normal_oof
    OptR = OptimizedRounder(n_overall=20, n_classwise=20)
    OptR.fit(raw_normal_oof, y_train)
    normal_oof_preds = OptR.predict(raw_normal_oof)
    truncated_result = truncated_cv_with_adjustment_of_distribution(
        normal_oof_preds, y_train, groups, test_nth_assessment, n_trials=1000)
    config["truncated_mean_adjust"] = truncated_result["mean"]
    config["truncated_std_adjust"] = truncated_result["std"]
    config["truncated_upper"] = truncated_result["0.95upper_bound"]
    config["truncated_lower"] = truncated_result["0.95lower_bound"]
    plot_confusion_matrix(
        y_train,
        normal_oof_preds,
        classes=np.array(["acc_0", "acc_1", "acc_2", "acc_3"]),
        normalize=True,
        save_path=output_dir / "confusion_matrix_normal_oof.png")

    # ===============================
    # === Save
    # ===============================
    save_path = output_dir / "output.json"
    save_json(config, save_path)
    np.save(output_dir / "oof_preds.npy", oof_preds)
    with open(output_dir / "model.pkl", "wb") as m:
        pickle.dump(models, m)
