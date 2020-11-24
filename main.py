import argparse

from sklearn.linear_model import LinearRegression  # Temporary, used for test
from sklearn.metrics import mean_squared_error, r2_score  # Temporary
from data_preparation import load_dataset, prepare_data, get_data_arrays
from data_evaluation import get_predictions_cv
from feature_engineering import (
    select_pca_features,
    select_forward_features,
    select_backward_features,
    select_polynomial_features,
    select_correlation_features,
)


def main():
    parser = argparse.ArgumentParser(
        description="Launch a regression pipeline given a dataset."
    )
    parser.add_argument(
        "dataset_filename",
        type=str,
        help="Path to the dataset's .csv file.",
    )
    parser.add_argument(
        "feature_selection",
        type=str,
        nargs="?",
        default=None,
        choices=["backward", "correlation", "forward", "pca", "polynomial"],
        help="Type of feature selection to apply.",
    )
    parser.add_argument(
        "--n_splits",
        "-n",
        type=int,
        nargs="?",
        default=2,
        help="Number of split for the cross-validation (default = 3).",
    )
    args = parser.parse_args()

    # Load the dataset and clean it
    dataset_filename = args.dataset_filename
    dataset_df = load_dataset(dataset_filename)
    prepare_data(dataset_df)
    print(f"+ Dataset {dataset_filename} loaded and cleaned")

    # Preprocess the data
    feature_selection = args.feature_selection
    if feature_selection == "backward":
        dataset_df = select_backward_features(dataset_df)
    elif feature_selection == "correlation":
        dataset_df = select_correlation_features(dataset_df)
    elif feature_selection == "forward":
        dataset_df = select_forward_features(dataset_df)
    elif feature_selection == "pca":
        dataset_df = select_pca_features(dataset_df)
    elif feature_selection == "polynomial":
        dataset_df = select_polynomial_features(dataset_df)
    print(f"+ Preprocessing {feature_selection} applied on data")

    # Chose the model
    model_name = "linear-regression"
    model = LinearRegression()
    print(f"+ Model {model_name} initialized \n")

    # Perform a cross validation
    n_splits = args.n_splits
    X, y_true = get_data_arrays(dataset_df)
    A = get_predictions_cv(X, y_true, model, n_splits=n_splits)
    X_train, X_test, Y_train, Y_test, Y_pred = A
    for i in range(len(X_train)):
        mse_round = mean_squared_error(Y_test[i], Y_pred[i])
        r2_round = r2_score(Y_test[i], Y_pred[i])
        print(
            f"[{i + 1}/{n_splits}]: Train set size: {X_train[i].shape[0]} / "
            f"Test set size: {Y_pred[i].shape[0]} / "
            f"MSE: {mse_round:2.2} - r2: {r2_round:2.2}"
        )


if __name__ == "__main__":
    main()
