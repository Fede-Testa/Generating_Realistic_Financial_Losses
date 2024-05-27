import torch
from NNs import *
from metrics import *
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class WGAN():
    def __init__(self, latent_dim, g_hidden_dim, d_hidden_dim,
                 lr, batch_size=64, latent_distr='normal', dim=4):
        """
        Initializes the WGAN model.

        Args:
            latent_dim (int): The dimension of the latent space.
            g_hidden_dim (int): The dimension of the hidden layer in the generator.
            d_hidden_dim (int): The dimension of the hidden layer in the discriminator.
            lr (float): The learning rate for the optimizer.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            latent_distr (str, optional): The distribution of the latent space. Defaults to 'normal'.
            dim (int, optional): The dimension of the output data. Defaults to 4.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device", self.device)      

        print('Loading WGAN...')

        self.latent_dim = latent_dim
        self.g_hidden_dim = g_hidden_dim
        self.dim = dim
        self.latent_distr = latent_distr
        self.lr = lr
        self.batch_size = batch_size

        self.G = torch.nn.DataParallel(
            Generator(latent_dim=latent_dim, 
                      g_hidden_dim=g_hidden_dim, 
                      g_output_dim=dim)
            ).to(self.device)

        self.D = torch.nn.DataParallel(
            Discriminator(d_input_dim=dim, d_hidden_dim=d_hidden_dim)
            ).to(self.device)

        print('WGAN loaded.')

    def __load_train_data(self, train_path):
        """
        Loads and preprocesses the training data.

        Args:
            train_path (str): The file path to the training data.

        Returns:
            torch.utils.data.DataLoader: The DataLoader object containing the preprocessed training data.
        """
        data_train = pd.read_csv(
            train_path, names=["idx", "X1", "X2", "X3", "X4"], header=0
            )
        data_train = data_train.set_index(["idx"])

        train = torch.tensor(
            data_train.values.astype(np.float32)
        ).to(self.device)

        self.means = train.mean(dim=0, keepdim=True)
        self.stds = train.std(dim=0, keepdim=True)
        train_normalized = (train - self.means) / self.stds
        train_dataset = data_utils.TensorDataset(train_normalized)
        train_loader = data_utils.DataLoader(dataset=train_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True)
        return train_loader
        
    def __load_val_data(self, val_path, to_tensor=False):
        """
        Load validation data from a CSV file.

        Args:
            val_path (str): The path to the CSV file containing the validation data.
            to_tensor (bool, optional): Whether to convert the loaded data to a PyTorch tensor. Defaults to False.

        Returns:
            pandas.DataFrame or torch.Tensor: The loaded validation data. If `to_tensor` is True, it returns a PyTorch tensor, otherwise it returns a pandas DataFrame.
        """
        val_data = pd.read_csv(val_path)
        val_data = val_data.set_index(["idx"])
        if to_tensor:
            val_data = torch.tensor(
                val_data.values.astype(np.float32)
            ).to(self.device)
        return val_data
        
    def train(self, train_path, epochs=2000, early_stopping=False,
              val_path=None, patience=20, checkpoints=True, every=500):
        """
        Trains the generative model.

        Args:
            train_path (str): The file path to the training data.
            epochs (int, optional): The number of training epochs. Defaults to 2000.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            val_path (str, optional): The file path to the validation data. Required if early_stopping is True. Defaults to None.
            patience (int, optional): The number of epochs to wait for improvement before stopping training. 
                Required if early_stopping is True. Defaults to 20.
            checkpoints (bool, optional): Whether to save checkpoints during training. 
                Required if val_path is None. Defaults to True.
            every (int, optional): The number of epochs between saving checkpoints. 
                Required if checkpoints is True. Defaults to 500.

        Returns:
            tuple or None: A tuple containing the lists of AD and AKE distances, and the best epoch if early stopping is used. 
            Returns None if neither early stopping nor checkpoints are used.
        """
        ad_list = []
        ake_list = []
        epoch_list = []
        # load data
        train_loader = self.__load_train_data(train_path)
        if early_stopping:
            best_epoch = -1
            patience_counter = 0
            best_distance = np.inf
            if val_path is None:
                raise ValueError(
                    "val_path must be provided when early_stopping is True."
                    )
            if patience is None:
                raise ValueError(
                    "patience must be provided when early_stopping is True."
                    )
        if val_path is not None:
            data_val = self.__load_val_data(val_path, to_tensor=True)
            n_comp = data_val.shape[0]  # number of comparison values
        if checkpoints and val_path is None:
            raise ValueError(
                "val_path must be provided when checkpoints is True."
                )

        # define optimizers
        G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)
        D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr,
                                    maximize=True)

        n_epoch = epochs
        for epoch in tqdm(range(1, n_epoch + 1)):
            with tqdm(enumerate(train_loader), total=len(train_loader),
                      leave=False, desc=f"Epoch {epoch}") as pbar:
                for _, x in pbar:
                    x = x[0]  # data is encapsulated in a list
                    # perform a step of the training with the current batch
                    self.__step_G(x, G_optimizer)
                    self.__step_D(x, D_optimizer)

            # check for early stopping at the end of the epoch
            if early_stopping:
                # generate data (unnormalized)
                x_gen = self.generate(n_comp, to_tensor=True)
                # compute the AD and AKE between x_gen and data_val
                distance = energy_distance(x_gen, data_val)
                # if either doesn't improve, increment the patience counter
                if distance >= best_distance:
                    patience_counter += 1
                # if both of them improves, reset the patience counter
                else:
                    patience_counter = 0
                    best_distance = distance
                    best_epoch = epoch
                    ad_list.append(
                        AD_distance(x_gen.detach().numpy(),
                                    data_val.detach().numpy())
                        )
                    ake_list.append(
                        Absolute_Kendall_error(x_gen.detach().numpy(),
                                               data_val.detach().numpy())
                        )

                # if the patience counter reaches the limit, stop training
                if patience_counter == patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break
        
            if checkpoints and epoch % every == 0:
                x_gen = self.generate(n_comp, to_tensor=True)
                ad_list.append(AD_distance(
                    x_gen.detach().numpy(), data_val.detach().numpy())
                    )
                ake_list.append(
                    Absolute_Kendall_error(x_gen.detach().numpy(),
                                           data_val.detach().numpy())
                    )
                epoch_list.append(epoch)

        print('Training done')

        if early_stopping:
            print(f"Best epoch: {best_epoch}")
            return ad_list, ake_list, best_epoch
        elif checkpoints:
            return ad_list, ake_list, epoch_list
        else:
            return None
        
    def generate(self, n, to_tensor=False):
        """
        Generates n samples from the model.

        Args:
            n (int): The number of samples to generate.
            to_tensor (bool, optional): Whether to return the generated samples as a PyTorch tensor. Defaults to False.

        Returns:
            torch.Tensor or numpy.ndarray: The generated samples. If `to_tensor` is True, it returns a PyTorch tensor, otherwise it returns a numpy array.
        """
        z = self.__sample_from(n)
        x_gen = self.G(z)
        x_gen = x_gen * self.stds + self.means
        if to_tensor:
            return x_gen
        else:
            return x_gen.detach().numpy()
        
    def __sample_from(self, n_samples):
        """
        Sample from the latent distribution.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: The generated samples.
        """
        if self.latent_distr == "normal":
            return torch.randn(n_samples, self.latent_dim).to(self.device)
        elif self.latent_distr == "exp":
            return torch.Tensor.exponential_(
                torch.zeros((n_samples, self.latent_dim)), 2
                ).to(self.device)
        elif self.latent_distr == "gamma":
            return torch.distributions.gamma.Gamma(
                torch.tensor([1.0]), torch.tensor([1.0])
                ).sample((n_samples, self.latent_dim))[:, :, 0].to(self.device)
        elif self.latent_distr == "uniform":
            return torch.rand(n_samples, self.latent_dim).to(self.device)
        elif self.latent_distr == "student":
            return torch.distributions.studentT.StudentT(
                torch.tensor([1.5])
                ).sample((n_samples, self.latent_dim))[:, :, 0].to(self.device)
        else:
            raise ValueError(
                "Invalid distribution type. Please choose from 'normal', 'exp', 'gamma', 'uniform', or 'student'."
                )

    def __step_D(self, x, D_optimizer):
        """
        Performs a single training step for the discriminator network.

        Args:
            x (torch.Tensor): Real input data.
            D_optimizer (torch.optim.Optimizer): Optimizer for the discriminator network.

        Returns:
            float: Discriminator loss value.

        """
        self.D.zero_grad()

        # train discriminator on real
        x_real = x
        x_real = x_real.to(self.device)

        D_output_real = self.D(x_real)

        # train discriminator on fake
        z = self.__sample_from(x.shape[0])
        x_fake = self.G(z)

        D_output_fake = self.D(x_fake)

        # gradient backprop & optimize ONLY D's parameters  
        D_loss = (D_output_real - D_output_fake).mean()
        D_loss.backward()
        D_optimizer.step()

        # ensuring Lipischitz
        for p in self.D.parameters():
            p.data = torch.clamp(p.data, -0.01, 0.01)

        return D_loss.data.item()

    def __step_G(self, x, G_optimizer):
        """
        Performs a single optimization step for the generator network.

        Args:
            x (torch.Tensor): Input data.
            G_optimizer (torch.optim.Optimizer): Optimizer for the generator network.

        Returns:
            float: Generator loss value.
        """
        self.G.zero_grad()

        z = self.__sample_from(x.shape[0])

        G_output = self.G(z)
        D_output = self.D(G_output)
        G_loss = - D_output.mean()

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        G_optimizer.step()

        return G_loss.data.item()
    
    def evaluate(self, data):
        """
        Evaluates the model on the provided data.
        """
        data = torch.tensor(data).to(self.device)
        x_gen = self.generate(data.shape[0], to_tensor=True)
        ad = AD_distance(data, x_gen)
        ake = Absolute_Kendall_error(data, x_gen)
        return ad, ake

    def save_model(self, path="models/w_gan.pth"):
        """
        Saves the model to the specified path.
        """
        torch.save(self.G.state_dict(), path)

    def load_model(self, path="models/w_gan.pth"):
        """
        Loads the model from the specified path.
        """
        self.G.load_state_dict(torch.load(path))

    def train_with_history(self, train_path, epochs=2000, early_stopping=False, val_path=None, patience=20):
        """
        Trains the WGAN model with history of AD and AKE metrics.
        I's to illustrate the behaviour of AD and AKE metrics during the training.

        Args:
            train_path (str): The path to the training data.
            epochs (int, optional): The number of training epochs. Defaults to 2000.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            val_path (str, optional): The path to the validation data. Required if early_stopping is True. Defaults to None.
            patience (int, optional): The number of epochs to wait for improvement before stopping training. Required if early_stopping is True. Defaults to 20.

        Returns:
            tuple: A tuple containing the lists of AD and AKE metrics, and the best epoch (if early stopping is used).
        """

        ad_list = []
        ake_list = []
        epoch_list = []
        # load data
        train_loader = self.__load_train_data(train_path)
        if early_stopping:
            best_epoch = -1
            patience_counter = 0
            best_distance = np.inf
            if val_path is None:
                raise ValueError("val_path must be provided when early_stopping is True.")
            if patience is None:
                raise ValueError("patience must be provided when early_stopping is True.")
        if val_path is not None:
            data_val = self.__load_val_data(val_path, to_tensor=True)
            n_comp = data_val.shape[0]  # number of comparison values

        # define optimizers
        G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)
        D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr, maximize=True)

        n_epoch = epochs
        for epoch in tqdm(range(1, n_epoch + 1)):
            with tqdm(enumerate(train_loader), total=len(train_loader),
                      leave=False, desc=f"Epoch {epoch}") as pbar:
                for _, x in pbar:
                    x = x[0]  # data is encapsulated in a list
                    # perform a step of the training with the current batch
                    self.__step_G(x, G_optimizer)
                    self.__step_D(x, D_optimizer)

            if epoch % 25 == 0:
                if val_path is not None:
                    x_gen = self.generate(n_comp, to_tensor=True)
                    ad_list.append(
                        AD_distance(x_gen.detach().numpy(), data_val.detach().numpy())
                        )
                    ake_list.append(
                        Absolute_Kendall_error(x_gen.detach().numpy(), data_val.detach().numpy())
                        )
                    epoch_list.append(epoch)

                # check for early stopping at the end of the epoch
                if early_stopping:
                    # generate data (unnormalized)
                    x_gen = self.generate(n_comp, to_tensor=True)
                    # compute the AD and AKE between x_gen and data_val
                    distance = energy_distance(x_gen, data_val)
                    # if either doesn't improve, increment the patience counter
                    if distance >= best_distance:
                        patience_counter += 1
                    # if both of them improves, reset the patience counter
                    else:
                        patience_counter = 0
                        best_distance = distance
                        best_epoch = epoch

                    # if the patience counter reaches the limit, stop training
                    if patience_counter == patience:
                        print(f"Early stopping at epoch {epoch}.")
                        break
        
        print('Training done')

        if early_stopping:
            print(f"Best epoch: {best_epoch}")
            return ad_list, ake_list, best_epoch
        else:
            return ad_list, ake_list, epoch_list
        
    def cross_val(self, train_path, epochs=2000, late_start_percentage=0.05):
        """
        Perform cross-validation for training a generative model using Wasserstein GAN.

        Args:
            train_path (str): The file path to the training data.
            epochs (int, optional): The number of training epochs. Defaults to 2000.
            late_start_percentage (float, optional): The percentage of epochs to inlcude when calculating metrics. 
                                                     Defaults to 0.05.

        Returns:
            ad_list (list): List of late start average AD (Absolute Difference) distances for each fold.
            ake_list (list): List of late start average AKE (Absolute Kendall Error) distances for each fold.
        """
        ad_list = []
        ake_list = []
        train_data = pd.read_csv(train_path)
        train_data = train_data.set_index(["idx"])
        train_data = train_data.values
        kf = KFold(n_splits=5)
        for train_index, val_index in kf.split(train_data):
            print("Fold: ", len(ad_list)+1)
            train = train_data[train_index]
            train = torch.tensor(
                train.astype(np.float32)
            ).to(self.device)
            val = train_data[val_index]

            late_ad = 0
            late_ake = 0

            self.means = torch.tensor(
                train.mean(axis=0, keepdims=True)
                ).to(self.device)
            self.stds = torch.tensor(
                train.std(axis=0, keepdims=True)
                ).to(self.device)

            train_normalized = (train - self.means) / self.stds
            train_dataset = data_utils.TensorDataset(
                torch.tensor(train_normalized)
                )
            train_loader = data_utils.DataLoader(dataset=train_dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

            # define optimizers
            G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)
            D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr,
                                        maximize=True)
            n_epoch = epochs
            for epoch in tqdm(range(1, n_epoch + 1)):
                with tqdm(enumerate(train_loader), total=len(train_loader),
                          leave=False, desc=f"Epoch {epoch}") as pbar:
                    for _, x in pbar:
                        x = x[0]
                        # perform a step of the training with the current batch
                        self.__step_G(x, G_optimizer)
                        self.__step_D(x, D_optimizer)

                # late start average of AD and AKE to handle noise from the metrics
                if epoch > (1-late_start_percentage)*n_epoch:
                    x_gen = self.generate(val.shape[0], to_tensor=True)
                    late_ad += AD_distance(x_gen.detach().numpy(), val)
                    late_ake += Absolute_Kendall_error(
                        x_gen.detach().numpy(), val
                        )

            ad_list.append(late_ad/(late_start_percentage*n_epoch))
            ake_list.append(late_ake/(late_start_percentage*n_epoch))

        return ad_list, ake_list
    
    # # unstable, do not use
    # def train_val_on_metrics(self, train_path, epochs=2000,
    #                          early_stopping=False, val_path=None, patience=20):

    #     # load data
    #     train_loader = self.__load_train_data(train_path)

    #     best_ad = np.inf
    #     best_ake = np.inf
    #     best_epoch = -1
    #     patience_counter = 0

    #     # validation data should simply be loaded to serve as a comparison with generated data after each epoch
    #     if early_stopping:
    #         if val_path is None:
    #             raise ValueError("val_path must be provided when early_stopping is True.")
    #         if patience is None:
    #             raise ValueError("patience must be provided when early_stopping is True.")
    #         data_val = self.__load_val_data(val_path)
    #         n_comp = data_val.shape[0]  # number of comparison values

    #     # define optimizers
    #     G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)
    #     D_optimizer = optim.RMSprop(self.D.parameters(), lr=self.lr, maximize=True)

    #     n_epoch = epochs
    #     for epoch in tqdm(range(1, n_epoch + 1)):
    #         with tqdm(enumerate(train_loader), total=len(train_loader),
    #                   leave=False, desc=f"Epoch {epoch}") as pbar:
    #             for batch_idx, x in pbar:
    #                 x = x[0]  # data is encapsulated in a list
    #                 # perform a step of the training with the current batch
    #                 self.__step_G(x, G_optimizer)
    #                 self.__step_D(x, D_optimizer)

    #         # check for early stopping at the end of the epoch
    #         if early_stopping:
    #             # generate data (unnormalized)
    #             x_gen = self.generate(n_comp)
    #             # compute the AD and AKE between x_gen and data_val
    #             ad = AD_distance(data_val, x_gen)
    #             ake = Absolute_Kendall_error(data_val, x_gen)
    #             # if either doesn't improve, increment the patience counter
    #             if ad >= best_ad or ake >= best_ake:
    #                 patience_counter += 1
    #             # if both of them improves, reset the patience counter
    #             else:
    #                 patience_counter = 0
    #                 best_ad = ad
    #                 best_ake = ake
    #                 best_epoch = epoch

    #             # if the patience counter reaches the limit, stop training
    #             if patience_counter == patience:
    #                 print(f"Early stopping at epoch {epoch}.")
    #                 break
        
    #     print('Training done') 

    #     if early_stopping:
    #         print(f"Best AD: {best_ad}, Best AKE: {best_ake}, Best epoch: {best_epoch}")
    #         return best_ad, best_ake, best_epoch
    #     else:
    #         return epochs # return the last epoch if early stopping is not used
    