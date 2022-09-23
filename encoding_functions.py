from pulp import *
import cplex

def define_number_of_outputs(model):
    inpt = []

    for l in ["firstlayer", "secondlayer"]:
        inpt.append(model.get_layer(l).input_shape[1])

    return inpt

def get_weights_and_bias(model, n_layers):
    A = []
    b = []

    for n_of_layer in range(n_layers):
        total_weights = model.layers[n_of_layer].get_weights()

        weights = total_weights[0]
        bias = total_weights[1]

        temp_layer_weights = []
        temp_bias_layer = []

        w = weights.transpose()

        for i in range(len(w)):
            temp_layer_weights.append(w[i])
            temp_bias_layer.append(bias[i])

        b.append(temp_bias_layer)
        A.append(temp_layer_weights)

    return A, b

def read_categorical_indexes(df_name):
    with open(".\datasets_boolean_index\\" + df_name + "\\" + df_name + "_categorical_indexes.txt") as f:
        ids = f.read().splitlines()

    return [int(v) for v in ids]

def define_integrity_constraints(s, y, n_of_layer):

    var_name = "z" + y.name.split("y", 1)[1]

    constraints = []
    z = LpVariable(var_name, lowBound=0, upBound=1, cat=LpBinary)

    M = sys.maxsize

    constraints.append(y <= M * (1 - z))
    constraints.append(s <= M * z)

    return constraints

def define_formula(categorical_ids, N_OF_LAYERS, A, b):

    pb = cplex.Cplex()

    def set_type(i):
        return "B" if i in categorical_ids else "C"

    def get_type(variables_to_type):
        return [v.split("_")[-1] for v in variables_to_type]

    def get_variables_names(id):
        suffix = '{0}_layer_{1}'.format(id, n_of_layer)
        return ["y" + suffix, "s" + suffix]

    def drop_elements_from_list(initial, to_remove):
        return [v for v in initial if v not in to_remove]

    def decouple_couple(var_index, weights):
        return [list(value) for value in zip(*[couple for couple in zip(var_index, weights)])]

    def set_indicator_constraint(formula, support_variables):
        import re
        form = [int(v) for v in re.findall(r'\d+', support_variables[0])]
        z_var = "z_{0}_layer_{1}".format(form[0], form[1])
        formula.variables.add(names=[z_var], types=["B"])
        idx = formula.variables.get_indices(support_variables)
        first = True
        for i in idx:
            flag = 0 if first else 1; first = False
            formula.indicator_constraints.add(indvar=z_var,
                                              lin_expr=[[i], [1]],
                                              sense="L",
                                              rhs=0,
                                              complemented=flag,
                                              indtype=formula.indicator_constraints.type_.if_)


    n_of_layer = 0

    for n_of_layer in range(N_OF_LAYERS):
        w_id = 0
        variables = \
            ['x_{0}_layer_{1}_type_{2}'.format(i, n_of_layer, set_type(i)) for i in range(len(A[n_of_layer][0]))]

        pb.variables.add(names=variables, lb=[0] * len(variables), types=get_type(variables))

        for i in range(len(A[n_of_layer])):
            weights = A[n_of_layer][i].tolist()
            weights.extend([-1, +1])

            rhs = -1 * b[n_of_layer][i]

            support_variables = get_variables_names(w_id)
            sup_vars_n = len(support_variables)
            pb.variables.add(names=support_variables, lb=[0] * sup_vars_n, types=["C"] * sup_vars_n)

            variables.extend(support_variables)
            var_ids = pb.variables.get_indices(name=variables)

            idx, w = decouple_couple(var_ids, weights)

            pb.linear_constraints.add(lin_expr=[[idx, w]],
                                      rhs=[rhs],
                                      names=["c{0}".format(i)])

            variables = drop_elements_from_list(variables, support_variables)

            set_indicator_constraint(pb, support_variables)

            w_id += 1

    pb.write("example.lp")

    return pb

def encoding_model(model, df_name):
    N_OF_LAYERS = 2
    A, b = get_weights_and_bias(model, N_OF_LAYERS)
    categorical_ids = read_categorical_indexes(df_name)
    inpt = define_number_of_outputs(model)
    pb = define_formula(categorical_ids, N_OF_LAYERS, A, b)

    return pb
