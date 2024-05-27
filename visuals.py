import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ecdfs(x_edm, x_wgan, x_real):
    """
    Plot the empirical cumulative distribution functions (ECDFs) for the given arrays.

    Parameters:
    x_edm (numpy.ndarray): Array containing the values for the EDM model.
    x_wgan (numpy.ndarray): Array containing the values for the WGAN model.
    x_real (numpy.ndarray): Array containing the real values.

    Returns:
    None
    """

    n = x_real.shape[0]
    n_plots = x_real.shape[1]
    plt.subplots(n_plots, 1, figsize=(10, 5*n_plots))

    for i in range(n_plots):
        plt.subplot(n_plots, 1, i+1)
        plt.plot(np.sort(x_edm[:,i]), np.arange(1, n+1)/n, label="edm")
        plt.plot(np.sort(x_wgan[:,i]), np.arange(1, n+1)/n, label="wgan")
        plt.plot(np.sort(x_real[:,i]), np.arange(1, n+1)/n, label="real")
        plt.legend()
        plt.title(f"ECDF for variable X{i+1}")
    plt.show()


def grid_plot(gen_data, real_data, quantile=None):
    """
    Generate a grid plot of the generated and real data.

    Parameters:
    - gen_data (numpy.ndarray): The generated data to be plotted.
    - real_data (numpy.ndarray): The real data to be plotted.
    - quantile (float, optional): The quantile value used to filter the data. Default is None.

    Returns:
    - None

    If quantile is None, the function will plot a pairplot of the generated and real data.
    If quantile is provided, the function will filter the data based on the quantile value and plot the filtered data.

    Note: This function requires the pandas and seaborn libraries to be installed.
    """

    gen_df = pd.DataFrame(gen_data, columns=["X1", "X2", "X3", "X4"])
    gen_df["type"] = "Generated"
    real_df = pd.DataFrame(real_data, columns=["X1", "X2", "X3", "X4"])
    real_df["type"] = "Real"
    df = pd.concat([gen_df, real_df])
    if quantile is None:
        # plot the pairplot
        sns.pairplot(df, hue="type", diag_kind="kde")
        plt.show()
    else:
        if "type" in gen_df.columns:
            gen_df.drop(columns="type", inplace=True)

        percent = gen_df.quantile(quantile)
        gen_df = gen_df[gen_df.apply(lambda x: any(x > percent), axis=1)]

        # repeat the same with real_df
        if "type" in real_df.columns:
            real_df.drop(columns="type", inplace=True)
        percent = real_df.quantile(quantile)
        real_df = real_df[real_df.apply(lambda x: any(x > percent), axis=1)]

        # add back the type columns, create df_95
        gen_df["type"] = "Generated"
        real_df["type"] = "Real"

        df = pd.concat([gen_df, real_df])

        # plot the pairplot
        sns.pairplot(df, hue="type", diag_kind="kde")

