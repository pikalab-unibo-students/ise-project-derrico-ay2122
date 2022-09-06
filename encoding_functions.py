import numpy as np
from pysmt.shortcuts import Symbol, Plus, Times, Real
from pysmt.typing import REAL

def define_number_of_outputs(model):
    outpt, inpt = [], []

    for l in ["firstlayer", "secondlayer"]:
        inpt.append(model.get_layer(l).input_shape[1])
        outpt.append(model.get_layer(l).output_shape[1])

    return inpt, outpt

def get_weights_and_bias(model, input_dims, output_dims):
    A = []
    b = []
    n_of_layer = 0
    for layer in zip(input_dims, output_dims):
        total_weights = model.layers[n_of_layer].get_weights()
        weights = total_weights[0]
        bias = total_weights[1]
        temp_layer_weights = []
        temp_bias_layer = []

        for output_node in range(layer[1]):
            row = []
            for input_node in range(layer[0]):
                 row.append(weights[input_node][output_node])
            temp_layer_weights.append(row)
            temp_bias_layer.append(bias[output_node])

        b.append(temp_bias_layer)
        A.append(temp_layer_weights)
        n_of_layer = n_of_layer + 1

    return A, b

def read_categorical_indexes(df_name):
    with open(".\datasets_boolean_index\\" + df_name + "\\" + df_name + "_categorical_indexes.txt") as f:
        ids = f.read().splitlines()

    return [int(v) for v in ids]

def define_vars(categorical_ids, input_dim, output_dim):
    enc = []
    for input_dimension in input_dim:
        layer = []
        for i in range(input_dimension):
            var_name = 'x{0}'.format(i + 1)
            layer.append(Symbol(var_name, typename=REAL))
            #layer.append(Symbol(var_name + "_real", typename=REAL) if not i in categorical_ids
                       #else Symbol(var_name + "_bool", typename=BOOL))

        enc.append(layer)

    return enc

def define_formula(variables, A, b):
    #print(variables)
    for n_of_layer in range(len(variables)):
        formula = None
        for output in A[n_of_layer]:
            constants = [Real(float(v)) for v in output]
            formula = Times(constants[0], variables[n_of_layer][0])
            for i in range(1, len(constants)):
                formula = Plus(formula, Times(constants[i], variables[n_of_layer][i]))
                print(formula)
            constraint_id = A[n_of_layer].index(output)
            #formula = Plus(formula, )
            #print(formula)


def encoding_model(model, df_name):
    inpt, outpt = define_number_of_outputs(model)
    A, b = get_weights_and_bias(model, inpt, outpt)
    ids = read_categorical_indexes(df_name)
    enc = define_vars(ids, inpt, outpt)
    define_formula(enc, A, b)
    print(enc)

    #for number_of_layer in range(len(A)):
     #   layer_weights = A[number_of_layer]
        #print(len(layer_weights))
