import torch
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from NNs import Generator
from metrics import energy_distance, AD_distance, Absolute_Kendall_error

class EDM():
    def __init__(self, latent_dim, g_hidden_dim, lr, batch_size=64, latent_distr='normal', dim=4):
        """
        Initialize the EDM (Event-driven Model) class.

        Args:
            latent_dim (int): The dimension of the latent space.
            g_hidden_dim (int): The dimension of the hidden layer in the generator.
            lr (float): The learning rate for the optimizer.
            batch_size (int, optional): The batch size for training. Defaults to 64.
            latent_distr (str, optional): The distribution of the latent space. Defaults to 'normal'.
            dim (int, optional): The dimension of the output data. Defaults to 4.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device", self.device)      

        print('Loading EDM...')

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

        print('Model loaded.')

    def __load_train_data(self, train_path):
        """
        Loads and preprocesses the training data.

        Args:
            train_path (str): The file path to the training data.

        Returns:
            None
        """
        # stores the mean and std var values of the training data, to apply inverse transformation to the generated data
        data_train = pd.read_csv(train_path, names=["idx", "X1", "X2", "X3", "X4"], header=0)
        data_train = data_train.set_index(["idx"])

        print(data_train.values)

        train = torch.tensor(
            data_train.values.astype(np.float32)
            ).to(self.device)

        self.means = train.mean(dim=0, keepdim=True)
        self.stds = train.std(dim=0, keepdim=True)
        train_normalized = (train - self.means) / self.stds
        train_dataset = data_utils.TensorDataset(train_normalized)
        self.train_loader = data_utils.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
            )
        
    def __load_val_data(self, val_path, to_tensor=False):
        val_data = pd.read_csv(val_path)
        val_data = val_data.set_index(["idx"])
        if to_tensor:
            val_data = torch.tensor(
                val_data.values.astype(np.float32)
            ).to(self.device)
        return val_data
        
    def train(self, train_path, epochs=2000, early_stopping=False, val_path=None, patience=20):
        """
        Trains the model using the provided training data.

        Args:
            train_path (str): The file path to the training data.
            epochs (int, optional): The number of training epochs. Defaults to 2000.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.
            val_path (str, optional): The file path to the validation data. Required if early_stopping is True.
            patience (int, optional): The number of epochs to wait for improvement before stopping training. 
                Required if early_stopping is True.

        Returns:
            Union[int, Tuple[float, float, int]]: If early_stopping is False, returns the None.
                If early_stopping is True, returns a tuple containing the best AD distance, AKE error, and the 
                epoch at which the best loss was achieved.
        """
        # load data
        self.__load_train_data(train_path)

        validation_loss = []
        training_loss = []
        best_loss = np.inf
        best_epoch = -1
        patience_counter = 0

        # validation data should simply be loaded to serve as a comparison with generated data after each epoch
        if early_stopping:
            if val_path is None:
                raise ValueError("val_path must be provided when early_stopping is True.")
            if patience is None:
                raise ValueError("patience must be provided when early_stopping is True.")
            data_val = self.__load_val_data(val_path, to_tensor=True)
            n_comp = data_val.shape[0]  # number of comparison values

        # define optimizers
        G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)

        n_epoch = epochs
        for epoch in range(1, n_epoch + 1):
            t_loss = 0
            counter = 0
            with tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                      leave=False, desc=f"Epoch {epoch}") as pbar:
                for batch_idx, x in pbar:
                    x = x[0]  # data is encapsulated in a list
                    b_loss = self.__step(x, G_optimizer)
                    t_loss += b_loss 
                    counter += 1
            t_loss /= counter  # average loss over the epoch       
            training_loss.append(t_loss)

            # check for early stopping at the end of the epoch
            if early_stopping:
                # generate data (unnormalized)
                x_gen = self.generate(n_comp, to_tensor=True)
                # compute the AD and AKE between x_gen and data_val
                loss = energy_distance(data_val, x_gen).item()
                validation_loss.append(loss)
                if loss >= best_loss:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    best_loss = loss
                    best_epoch = epoch
                    ad = AD_distance(
                        data_val.detach().numpy(),
                        x_gen.detach().numpy()
                        )
                    ake = Absolute_Kendall_error(
                        data_val.detach().numpy(),
                        x_gen.detach().numpy()
                        )

                if patience_counter == patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break
        
        print('Training done')

        if early_stopping:
            print(f"Best loss: {best_loss}, Best epoch: {best_epoch}, AD distance: {ad}, AKE error: {ake}")
            return ad, ake, best_epoch
        else:
            return None
        
    def __sample_from(self, n_samples):
        """
        Samples from the specified latent distribution.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: A tensor containing the generated samples.

        Raises:
            ValueError: If an invalid distribution type is specified.

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

    def generate(self, n, to_tensor=False):
        """
        Generates n samples from the model.
        """
        z = self.__sample_from(n)
        x_gen = self.G(z)
        x_gen = x_gen * self.stds + self.means
        if to_tensor:
            return x_gen
        else:
            return x_gen.detach().numpy()

    def __step(self, x, G_optimizer):
        """
        Performs a single step of the training of the generator.
        Returns the loss value.
        """
        self.G.zero_grad()
        # sample from the latent space
        z = self.__sample_from(x.shape[0])
        # generate data
        x_gen = self.G(z)
        # calculate the energy distance
        loss = energy_distance(x, x_gen)
        # backpropagate
        loss.backward()
        G_optimizer.step()
        return loss.item()
    
    def evaluate(self, data):
        """
        Evaluates the model on the provided data.
        """
        data = torch.tensor(data).to(self.device)
        x_gen = self.generate(data.shape[0], to_tensor=True)
        ad = AD_distance(data, x_gen)
        ake = Absolute_Kendall_error(data, x_gen)
        return ad, ake
  
    def save_model(self, path="models/energy_distance_model.pth"):
        """
        Saves the model to the specified path.
        """
        torch.save(self.G.state_dict(), path)

    def load_model(self, path="models/energy_distance_model.pth"):
        """
        Loads the model from the specified path.
        """
        self.G.load_state_dict(torch.load(path))
    
    def train_val_on_metrics(self, train_path, epochs=2000, early_stopping=False, val_path=None, patience=20):
        """
        Trains the model using the provided training data and validates it on the provided validation data.
        Validation is done on the AD distance and AKE error.
        Usage is not not recommended, since this early stopping criterion is not very reliable.
        """
        # load data
        self.__load_train_data(train_path)

        best_ad = np.inf
        best_ake = np.inf
        best_epoch = -1
        patience_counter = 0

        # validation data should simply be loaded to serve as a comparison with generated data after each epoch
        if early_stopping:
            if val_path is None:
                raise ValueError(
                    "val_path must be provided when early_stopping is True."
                    )
            if patience is None:
                raise ValueError(
                    "patience must be provided when early_stopping is True."
                    )
            data_val = self.__load_val_data(val_path)
            n_comp = data_val.shape[0]  # number of comparison values

        # define optimizers
        G_optimizer = optim.RMSprop(self.G.parameters(), lr=self.lr)

        n_epoch = epochs
        for epoch in range(1, n_epoch + 1):
            with tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                      leave=False, desc=f"Epoch {epoch}") as pbar:
                for batch_idx, x in pbar:
                    x = x[0]  # data is encapsulated in a list
                    # perform a step of the training with the current batch
                    self.__step(x, G_optimizer)

            # check for early stopping at the end of the epoch
            if early_stopping:
                # generate data (unnormalized)
                x_gen = self.generate(n_comp)
                # compute the AD and AKE between x_gen and data_val
                ad = AD_distance(data_val, x_gen)
                ake = Absolute_Kendall_error(data_val, x_gen)
                # if either doesn't improve, increment the patience counter
                if ad >= best_ad or ake >= best_ake:
                    patience_counter += 1
                # if both of them improves, reset the patience counter
                else:
                    patience_counter = 0
                    best_ad = ad
                    best_ake = ake
                    best_epoch = epoch

                if patience_counter == patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break
        
        print('Training done') 

        if early_stopping:
            print(f"Best AD: {best_ad}, Best AKE: {best_ake}, Best epoch: {best_epoch}")
            return best_ad, best_ake, best_epoch
        else:
            return None
