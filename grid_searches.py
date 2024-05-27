import itertools
import pandas as pd
from w_gan import WGAN
from edm import EDM


def grid_search_wgan(train_path, val_path, par_dict, epochs, every, path):
    """
    Perform a grid search for WGAN models.

    Args:
        train_path (str): Path to the training data.
        val_path (str): Path to the validation data.
        par_dict (dict): Dictionary of hyperparameters to search over.
        epochs (int): Number of epochs to train each model.
        every (int): Number of epochs between checkpoints.
        path (str): Path to save the results.

    Returns:
        pandas.DataFrame: DataFrame containing the results of the grid search.

    """
    results = pd.DataFrame(
        columns=list(par_dict.keys()) + ["AD", "AKE", "epochs"]
    )
    index = 0
    for par_comb in itertools.product(*par_dict.values()):
        # create a dictionary with the hyperparameters
        par = dict(zip(par_dict.keys(), par_comb))

        # create a model
        model = WGAN(**par)

        # print model parameters
        print(f"WGAN, parameters: {par}")

        # train the model
        ad_list, ake_list, epoch_list = model.train(
            train_path=train_path, epochs=epochs, checkpoints=True,
            every=every, val_path=val_path
        )

        # store the results as additional rows of results, one for each value of the list
        for i in range(len(ad_list)):
            new_row = pd.DataFrame({**par, "AD": ad_list[i],
                                    "AKE": ake_list[i], "epochs": epoch_list[i]}, index=[index])
            index += 1
            results = pd.concat([results, new_row], axis=0)

    results.to_csv(path)
    return results


def grid_search_edm(train_path, val_path, par_dict, epochs, path, patience=100):
    """
    Perform a grid search for EDM models.

    Args:
        train_path (str): Path to the training data.
        val_path (str): Path to the validation data.
        par_dict (dict): Dictionary of hyperparameters to search over.
        epochs (int): Number of epochs to train each model.
        path (str): Path to save the results.
        patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to 100.

    Returns:
        pandas.DataFrame: DataFrame containing the results of the grid search.

    """
    results = pd.DataFrame(columns=list(par_dict.keys()) + ["AD", "AKE", "epochs"])

    index = 0

    for par_comb in itertools.product(*par_dict.values()):
        # create a dictionary with the hyperparameters
        par = dict(zip(par_dict.keys(), par_comb))

        # create a model
        model = EDM(**par)

        # print which model
        print(f"EDM, number: {index+1}, parameters: {par}")

        # train the model
        ad, ake, val_epoch = model.train(
            train_path=train_path, val_path=val_path, epochs=epochs,
            early_stopping=True, patience=patience
            )

        # store the results as additional row of results
        new_row = pd.DataFrame({**par, "AD": ad, "AKE": ake,
                                "epochs": val_epoch}, index=[index])
        results = pd.concat([results, new_row])
        index += 1

    # save the results
    results.to_csv(path)

    return results
