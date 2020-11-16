import argparse

from data_preparation import load_dataset, prepare_data


def main():
    parser = argparse.ArgumentParser(
        description="Launch a regression pipeline given a dataset."
    )
    parser.add_argument(
        "dataset_filename",
        type=str,
        help="Path to the dataset's .csv file.",
    )
    args = parser.parse_args()

    dataset_df = load_dataset(args.dataset_filename)
    print(dataset_df)
    prepare_data(dataset_df)
    print(dataset_df)


if __name__ == "__main__":
    main()
