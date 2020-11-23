import argparse

from sklearn.linear_model import LinearRegression  # Temporary, used for test

from data_preparation import load_dataset, prepare_data, get_data_arrays
from data_evaluation import get_predictions_cv


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
        "--n_splits",
        "-n",
        type=int,
        nargs="?",
        default=3,
        help="Number of split for the cross-validation (default = 3).",
    )
    args = parser.parse_args()

    dataset_df = load_dataset(args.dataset_filename)
    print(dataset_df)
    prepare_data(dataset_df)
    print(dataset_df)
    X, y_true = get_data_arrays(dataset_df)
    print(X.shape, y_true.shape)
    model = LinearRegression()
    A = get_predictions_cv(X, y_true, model, n_splits=args.n_splits)
    for i in range(len(A[0])):
        print(
            f"[{i}/{args.n_splits}]: Train set size: {A[0][i].shape[0]} / "
            f"Test set size: {A[-1][i].shape[0]}"
        )


if __name__ == "__main__":
    main()
