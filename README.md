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

Every dataset is associated to a file which contains the indices of the categorical and boolean columns, that are used during the NN encoding phase.
These files are in ***datasets_categorical_index*** folder.


