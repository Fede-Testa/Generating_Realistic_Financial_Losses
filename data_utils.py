import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, destination_path1, destination_path2):
    """
    Load data from a CSV file, split it into train and test sets, and save them to separate CSV files.

    Args:
        path (str): The path to the input CSV file.
        destination_path1 (str): The path to save the train data CSV file.
        destination_path2 (str): The path to save the test data CSV file.

    Returns:
        None
    """
    data = pd.read_csv(path, names=["idx", "X1", "X2", "X3", "X4"], header=0)
    data = data.set_index(["idx"])

    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data.to_csv(destination_path1)
    test_data.to_csv(destination_path2)

    return


def load_data_balanced(path, destination_path1, destination_path2, percent=0.9):
    """
    Load data from a CSV file, balance the data based on a given percentile,
    split it into train and test sets, and save the results to CSV files.

    Args:
        path (str): The path to the input CSV file.
        destination_path1 (str): The path to save the train data CSV file.
        destination_path2 (str): The path to save the test data CSV file.
        percent (float, optional): The percentile to use for balancing the data.
            Defaults to 0.9.

    Returns:
        None
    """

    data = pd.read_csv(path, names=["idx", "X1", "X2", "X3", "X4"], header=0)
    data = data.set_index(["idx"])

    # Add label column
    quantile = data.quantile(percent)
    data['label'] = (data > quantile).any(axis=1)

    # Split the data into two dataframes
    data_true = data[data['label']]
    data_false = data[~data['label']]

    # Perform train_test_split on each dataframe
    train_data_true, test_data_true = train_test_split(
        data_true, test_size=0.2
        )
    train_data_false, test_data_false = train_test_split(
        data_false, test_size=0.2
        )

    # Concatenate the results, drop the label column
    train_data = pd.concat([train_data_true, train_data_false])
    train_data.drop(columns=['label'], inplace=True)
    test_data = pd.concat([test_data_true, test_data_false])
    test_data.drop(columns=['label'], inplace=True)

    # Save to csv
    train_data.to_csv(destination_path1)
    test_data.to_csv(destination_path2)

    return
