Implementing Wasserstein GAN (https://arxiv.org/abs/1701.07875) and a simple generative model based on energy distance for generating extreme value data of stock prices log returns.

Directory structure:

* *main.ipynb*: summary of the problem, then all the implementative details and performance assessments
* *Generative_Models.pdf*: description of the problem and the data, and a theoretical explanation of the models and the metrics used
* *edm.py* and *w_gan.py*: classes for the generative models
* *NNs.py*: classes for the neural networks employed by the models
* *metrics.py*: all distances and metrics used in the project
* *data_utils.py*: functions to load data
* *visuals.py*: functions for plots used for qualitative assessment of the models

The notebook *main.ipynb* is meant to guide through the project. We suggest to start from *main.ipynb* and use *Generative_Models.pdf* for any further inquiry on the topics, models or other aspects that are overlooked in the notebook.