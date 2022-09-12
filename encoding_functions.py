import numpy as np
from pulp import *

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
    print(model.summary())
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
    number_of_layer = 1
    for input_dimension in input_dim:
        layer = []
        for i in range(input_dimension):
            var_name = 'x{0}_layer{1}'.format(i, number_of_layer)
            x = LpVariable(name=var_name)
            layer.append(x)
            #layer.append(Symbol(var_name, typename=REAL))
            #layer.append(Symbol(var_name + "_real", typename=REAL) if not i in categorical_ids
                       #else Symbol(var_name + "_bool", typename=BOOL))

        enc.append(layer)
        number_of_layer = number_of_layer + 1

    return enc

def define_integrity_constraints(s, y):

    var_name = "z" + y.name.split("y", 1)[1]
    constraints = []
    z = LpVariable(var_name, 0, 1, cat='Integer')
    M = sys.maxsize

    constraints.append(y <= (1 - z) * M)
    constraints.append(s <= z * M)
    constraints.append(y >= 0)
    constraints.append(s >= 0)

    return constraints

def define_objective_function(pb):

    def split_name(variable):
        return variable.name.split("_")

    layers = list(set([split_name(v)[1] for v in pb.variables()]))
    layers.sort()

    def get_variables(var_name, layer):
        return [v for v in pb.variables() if (var_name in split_name(v)[0]) & (layer in split_name(v)[1])]

    def sort_variables(variables):
        sorted(variables, key=lambda v: v.name)

    def sum_variables(vars):
        objective_function = None
        n_vars = len(vars)
        for i in range(n_vars):
           objective_function += vars[i]

        return objective_function

    ys_part, zs_part = None, None

    j = 0

    for l in layers:
        j = j + 1
        ys = get_variables("y", l)
        zs = get_variables("z", l)
        sort_variables(ys), sort_variables(zs)
        ys_part += sum_variables(ys)
        zs_part += sum_variables(zs)

    return lpSum([ys_part, zs_part])

def define_formula(variables, A, b):

    pb = pulp.LpProblem("MILP encoding for Machine Learning problem")

    # per i 2 layer
    for n_of_layer in range(len(variables)):
        #per ogni input
        for i in range(len(A[n_of_layer])):
            formula = None
            #per ogni variabile
            for j in range(len(A[n_of_layer][i])):
                formula += A[n_of_layer][i][j] * variables[n_of_layer][j]

            formula += b[n_of_layer][i]

            def right_variable(name):
                return LpVariable(name=name + "{0}_layer{1}".format(i, n_of_layer + 1))

            y = right_variable("y")
            s = right_variable("s")

            formula = formula == y - s
            pb += (formula, "Layer {0} constraint {1}".format(n_of_layer, i))
            constraints = define_integrity_constraints(s, y)

            j = 0
            for c in constraints:
                pb += (c, "Layer {0} integrity constraint {1}-{2}".format(n_of_layer, i, j))
                j = j + 1

    pb.objective = define_objective_function(pb)
    print(pb)

    return None

def encoding_model(model, df_name):
    inpt, outpt = define_number_of_outputs(model)
    print(inpt)
    A, b = get_weights_and_bias(model, inpt, outpt)
    ids = read_categorical_indexes(df_name)
    enc = define_vars(ids, inpt, outpt)
    define_formula(enc, A, b)

    #print("const: ", len(constraints))
    #print("constraint[0]: ", len(constraints[0]))

    #print("blocks: ", len(blocks))
    #print("first: ", len(blocks[0]))
    #print("second: ", len(blocks[1]))
    #print(formula[1])
    #print("Is sat? ", is_sat(formula=formula[0], solver_name="mst"))

