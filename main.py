import ctypes.wintypes
import getopt
import os

from encoding_functions_SMT import print_solver_assertion, cancel_file
from explain import encoding_model, explaining_procedures

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import pandas as pd
from tensorflow import keras

from neural_net_functions import build_model, dataset_preprocessing

def parse_options():
    """
        Parses command-line options:
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'n:d:x:s:h',
                                   ['hidden_node=',
                                    'dataframe=',
                                    'solver=',
                                    'save',
                                    'help'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err).capitalize())
        usage()
        sys.exit(1)

    hidden_nodes = 20
    df_name = None
    save = False
    solver = "cplex"

    for opt, arg in opts:
        if opt in ('-n', '--hidden_node'):
            hidden_nodes = int(arg)
        elif opt in ('-d', '--dataframe'):
            df_name = str(arg)
        elif opt == ('--solver'):
            solver = str(arg)
        elif opt == ('--save'):
            save = True
        elif opt in ('-h', '--help'):
            usage()
            sys.exit(0)
        else:
            assert False, 'Unhandled option: {0} {1}'.format(opt, arg)

    return hidden_nodes, df_name, save, solver


def usage():
    """
        Prints usage message.
    """

    print('Usage:', os.path.basename(sys.argv[0]), '[options] lp-file')
    print('Options:')
    print('        -n, --hidden_node=<int>  Number of hidden node for the NN')
    print('                                 Available values: [10, 15, 20] (default: 20)')
    print('        --save                   Save the NN model')
    print('        -d, --dataframe=<string> Name of dataframe to explain')
    print('        --solver=<string>        Approach to get explanation')
    print('                                 Available values: [cplex, smt] (default: cplex)')
    print('        -h, --help')

def import_dataframe(df_name):
    return pd.read_csv(".\datasets_files\\" + df_name, index_col=[0], sep=',', na_values=[''])

def print_vars(vars):
    for v in vars:
        print(v.name + ": " + str(v.varValue))

def print_estimates(values, measure):
    print("Avg ", measure, " size: ", np.average(values))
    print("Min ", measure, " size: ", np.min(values))
    print("Max ", measure, " size: ", np.max(values))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    hidden_nodes, name, save, solver_name = parse_options()

    df = import_dataframe(name)
    df = dataset_preprocessing(df, name)

    explanations = []
    computational_times = []

    try:
        model = keras.models.load_model(".\\models\\" + name
                                    + "_hidden_" + str(hidden_nodes))
    except:
        model = build_model(df, hidden_nodes, name)
        model.save(".\\models\\" + name + "_hidden_" + str(hidden_nodes))

    encoded_model, input_vars = encoding_model(model, name, solver_name)

    for i in range(df.shape[0]):
        x_values = df.drop(columns="target").iloc[i, :]
        input_values = x_values.array.astype('float32').reshape(1, -1)
        print("Input values: ", input_values)

        correct_output = df['target']
        output_values = model.predict(input_values)[0]
        #
        output_values = [0 if i != output_values.tolist().index(max(output_values)) else 1 for i in range(len(output_values))]
        output_vars = ["X_{0}_layer_2_type_C".format(i) for i in range(len(output_values))]

        print("Prediction: ", output_values, " id predicted_class: ", output_values.index(1))

        len_expl, time = explaining_procedures(encoded_model, solver_name, (input_vars, input_values[0]), (output_vars, output_values))
        explanations.append(len_expl)
        computational_times.append(time)
        print("Sum: ", np.sum(computational_times))

    print_estimates(explanations, "Explanation")
    print_estimates(computational_times, "Computational time")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
