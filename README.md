# Abduction-Based-Explanations-for-ML-Models

Reproduction of the study "Alexey Ignatiev, Nina Narodytska, Joao Marques-Silva. *Abduction-Based Explanations for Machine Learning Models*"

## Getting Started

The following packages are necessary to run the code:

* [numpy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [cplex](https://pypi.org/project/cplex/)
* [pySMT](https://github.com/pysmt/pysmt) (with Z3 installed)
* [pySAT](https://github.com/pysathq/pysat)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [keras](https://pypi.org/project/keras/)
* [tensorflow](https://www.tensorflow.org/)
* [tensorflow_docs](https://github.com/tensorflow/docs)

## Usage
The script has a number of parameters, which can be set from the command line. To see the list of options, run (the executable script is located in src):
```
$ main.py -h
```

### Data reading and preprocessing

The script uses datasets in the CSV format: they are in the ***dataset_files*** path.
After the reading, the datas are preprocessed; these are the steps:

1. The names of the columns of the datasets contain indications on the type of features and they are used for the One Hot Encoding of the categorical features.
2. The ***target*** column with the indication of the class is LabelEncoded with value between 0 and n_classes-1.

Every dataset is associated to a file which contains the indices of the categorical and boolean columns, indices that are used during the NN encoding phase.
These files are in ***datasets_categorical_index*** folder.

MNIST Dataset is further elaborated: after the classes labelling, every pixel of the images is forced to be only white or black, because only 0 and 1 values are allowed; every intermediate values are properly rounded.

### Training a NN
Before extracting explanations, a Neural Net model must be trained:
```
$ main.py -n 𝑛𝑢𝑚𝑏𝑒𝑟_𝑜𝑓_𝑛𝑜𝑑𝑒 [...]
```
Here, a Neural Net with one hidden layer with ***number_of_node*** nodes is trained, if not already present in the ***models*** path. So, the first time, the created model is trained and saved in "models/***dataset_name***_***number_of_node***" folder, and then it becomes available for reproducing the experiment for a second time.

### Computing a formal explanation
A rigorous abduction-based explanation for the samples of a dataset can be computed by running the following command:
```
$ main.py -n 𝑛𝑢𝑚𝑏𝑒𝑟_𝑜𝑓_𝑛𝑜𝑑𝑒 -d 𝑑𝑎𝑡𝑎𝑠𝑒𝑡_𝑛𝑎𝑚𝑒 -e 𝑡𝑦𝑝𝑒_𝑜𝑓_𝑒𝑥𝑝𝑙𝑎𝑛𝑎𝑡𝑖𝑜𝑛 -s 𝑡𝑦𝑝𝑒_𝑜𝑓_𝑠𝑜𝑙𝑣𝑒𝑟_𝑢𝑠𝑒𝑑
```

It's possible to set a precise number of nodes for the hidden layer using the parameter ``` -e ```, as said above: for the experiment they have been considered NN with i ∈ {10, 15, 20} neurons.

With the parameter ```-d``` a dataset, whose samples will be explained, can be choosen.
This is the list of considered datasets, from Penn & UCI repositories of benchmark dataset:
* australian
* auto
* backache
* breast_cancer
* cleve
* cleveland
* glass
* glass2
* heart_statlog
* spect
* voting
* MNIST dataset
