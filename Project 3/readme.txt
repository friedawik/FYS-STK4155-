This is the project space for Project 3 of the FYS-STK4155 course. 

Some thoughts about the project: This was my first encounter with a Convolutional Neural Network. Since I will be using CNN models and explainable AI methods in my master project, I wanted to get some insights of the different parts of the model by visualising them. It was fun and interesting to see how different pattern evolved, but I learned that simple visualisations are not sufficient to make the model interpretable. For that, I hope some of the explainable AI methods that will be covered in the spring course "IN4310 – Deep Learning for Image Analysis" can be more suitable.

The following files are included in this repository:
project_3.pdf: The project report
animations.ipynb: Contains the animations and plots of the various CNN layers
plots.ipynb: Contains all other plots
pytorch_cnn.py: Contains the CNN model and some functions for training and testing the data.
pytorch_ffnn.py: Contains the FFNN model and the grid search for learning rate and regularisation.
pytorch_gridsearch_hyperparam: Contains the gridsearch loops used for the CNN model.
pytorch_gridsearch_arch.py: Contains a search over different kernel sizes, pooling sizes and strides. This search was performed but the results were not included in the report due to time limitations.
Visualizing Convolutional Layers in PyTorch.txt: Downloaded example from GPT UiO chat


The following folders are included in this repository:
cnn_models: Contains all CNN models saved during training
ffnnn_models: Contains all FFNN models saved during training

