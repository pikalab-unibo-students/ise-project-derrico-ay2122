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

It's possible to set a precise number of nodes for the hidden layer using the parameter ```-n```, as said above: for the experiment they have been considered NN with i ∈ {10, 15, 20} neurons.

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

```-e``` parameter let decide what kind of explanation realize, between subset- and cardinality-minimal explanations, following Algorithm 1 and Algorithm 2 respectively

```-s``` parameter is releated to the two possible solvers which can be used as oracles for the experiment: Cplex and SMT solvers.

### Reproducing the results
Working with MNIST Dataset means to have the possibility of visualizing the computed explanations. 
The following code draws the image rapresented by a sample from MNIST dataset with the explanation of the prediction made by the classifier: the original image is drawn with the white number on the black background, with the red pixels which show how the model sees the image, in order to make its prediction.
```python
def save_images(sample, expl, pattern_id):

        # image size
        sz = int(math.sqrt(len(sample)))

        light_blue_rgba = tuple([0, 255, 255, 230.0])
        white_rgba = tuple([255, 255, 255, 255.0])
        red_rgba = tuple([186, 6, 6, 255.0])
        black_rgba = tuple([0, 0, 0, 255.0])

        # original image
        pixels1, pixels2 = [], []  # this will contain an array of masked pixels
        for i in range(sz):
            row1, row2 = [], []
            for j, v in enumerate(sample[(i * sz):(i + 1) * sz]):
                id_pixel = i * sz + j

                if v == 1:
                    if id_pixel in expl:
                        row1.append(light_blue_rgba)
                    else:
                        row1.append(white_rgba)

                    row2.append(white_rgba)
                else:
                    if id_pixel in expl:
                        row1.append(red_rgba)
                    else:
                        row1.append(black_rgba)

                    row2.append(black_rgba)

            pixels1.append(row1)
            pixels2.append(row2)

        pixels1 = np.asarray(pixels1, dtype=np.uint8)
        pixels2 = np.asarray(pixels2, dtype=np.uint8)
        mpimg.imsave('.\\images\\sample{0}-patch.png'.format(pattern_id), pixels1, cmap=mpcm.gray, dpi=5)
        mpimg.imsave('.\\images\\sample{0}-orig.png'.format(pattern_id), pixels2, cmap=mpcm.gray, dpi=5)
```

## Citations

This work is born as a reproduction of the above quoted study, but this is not the only material used.
Here some "hot stuff":

```
@inproceedings{inms-neurips19,
  author    = {Alexey Ignatiev and
               Nina Narodytska and
               Joao Marques-Silva},
  title     = {On Relating Explanations and Adversarial Examples},
  booktitle = {NeurIPS},
  year      = {2019}
}

@inproceedings{iisms-aaai22a,
  author    = {Alexey Ignatiev and
               Yacine Izza and
               Peter J. Stuckey and
               Joao Marques-Silva},
  title     = {Using MaxSAT for Efficient Explanations of Tree Ensembles},
  booktitle = {AAAI},
  year      = {2022},
}
