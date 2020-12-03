import argparse

from data_preparation import load_dataset, prepare_data, get_data_arrays
from data_evaluation import get_predictions_cv, get_score_cv
from feature_engineering import (
    select_pca_features,
    select_forward_features,
    select_backward_features,
    select_polynomial_features,
    select_correlation_features,
)
from models import (
    linear_regression,
    lasso_regression,
    ridge_regression,
    elastic_net_regression,
)

POSSIBLE_MODELS = [
    "linear",
    "lasso",
    "ridge",
    "elastic-net",
    "backward",
    "forward",
    "polynomial", ]


def main():
    parser = argparse.ArgumentParser(
        description="launch a regression pipeline given a dataset"
    )
    parser.add_argument(
        "dataset_filename",
        type=str,
        help="path to the dataset's .csv file",
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=["all"] + POSSIBLE_MODELS,
        help="type of model to use. If 'all' is selected, do the evaluation on"
             "all models",
    )
    parser.add_argument(
        "-f",
        type=str,
        default=None,
        choices=["correlation", "pca"],
        help="type of feature selection to apply (default=None)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=2,
        metavar="",
        help="number of split for the cross-validation (default=3)",
    )
    args = parser.parse_args()

    # Load the dataset and clean it
    dataset_filename = args.dataset_filename
    dataset_df = load_dataset(dataset_filename)
    prepare_data(dataset_df)
    print(
        f"Dataset {dataset_filename} loaded and cleaned "
        f"({dataset_df.shape[0]} samples)"
    )

    # Select the features
    feature_selection = args.f
    if feature_selection:
        if feature_selection == "correlation":
            dataset_df = select_correlation_features(dataset_df)
        if feature_selection == "pca":
            dataset_df = select_pca_features(dataset_df)
        print(f"Preprocessing {feature_selection} applied on data")
    model_name = args.model_name
    n_splits = args.n
    if model_name == "all":
        for model in POSSIBLE_MODELS:
            run_model(dataset_df, model, n_splits, detailed_log=False)
    else:
        run_model(dataset_df, model_name, n_splits)


def run_model(data, model_name, n_splits, detailed_log=True):
    # Chose the model to use
    if model_name == "linear":
        model = linear_regression()
    if model_name == "lasso":
        model = lasso_regression()
    if model_name == "ridge":
        model = ridge_regression()
    if model_name == "elastic-net":
        model = elastic_net_regression()
    if model_name == "backward":
        data = select_backward_features(data)
        model = linear_regression()
    if model_name == "forward":
        data = select_forward_features(data)
        model = linear_regression()
    if model_name == "polynomial":
        data = select_polynomial_features(data)
        model = linear_regression()
    if detailed_log:
        print(f"Model {model_name} initialized")

    # Perform a cross validation
    X, y_true = get_data_arrays(data)
    A = get_predictions_cv(X, y_true, model, n_splits=n_splits)
    X_train, X_test, Y_train, Y_test, Y_pred = A
    if detailed_log:
        for i in range(len(X_train)):
            print(
                f"[{i + 1}/{n_splits}]: Train set size: {X_train[i].shape[0]} "
                f"/ Test set size: {Y_pred[i].shape[0]}"
            )

    # Compute cross-validation, median, mean and standard deviation MSE and r2
    print(f"{model_name.capitalize()} model results :")
    print(get_score_cv(Y_pred, Y_test) + "\n")


if __name__ == "__main__":
    main()
