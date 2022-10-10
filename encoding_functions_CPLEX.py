import os
import sys
import cplex
from encoding_utils_functions import generate_variables, get_variables_names


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
    path = ".\datasets_boolean_index\\" + df_name + "\\" + df_name + "_categorical_indexes.txt"

    ids = []
    if os.path.exists(path):
        with open(path) as f:
            ids = f.read().splitlines()
        ids = [int(v) for v in ids]

    return ids


def set_indicator_constraint(formula, support_variables):
    import re
    form = [int(v) for v in re.findall(r'\d+', support_variables[0])]
    var_id, n_layer = form[0], form[1]
    z_var = "Z_{0}_layer_{1}".format(var_id, n_layer)
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
                                          name="ic_{0}_layer_{1}".format(var_id, n_layer),
                                          indtype=formula.indicator_constraints.type_.if_)


def get_vars(formula, name):
    vars = [v for v in formula.variables.get_names() if name[0] in v] if len(name) == 1 \
        else [v for v in formula.variables.get_names() if name[0] in v and name[1] in v]
    print("VARS: ", vars)
    return vars


def get_type(variables_to_type):
    return [v.split("_")[-1] for v in variables_to_type]


def drop_elements_from_list(initial, to_remove):
    return [v for v in initial if v not in to_remove]


def decouple_couple(var_index, weights):
    return [list(value) for value in zip(*[couple for couple in zip(var_index, weights)])]


def freeze_input_and_output_CPLEX(pb, inputs, outputs):

    input_vars, input_values = inputs
    output_vars, output_values = outputs

    hypos = freeze_input_CPLEX(pb, input_vars, input_values)
    freeze_output_CPLEX(pb, output_vars, output_values)

    return hypos


def freeze_input_CPLEX(pb, input_vars, input_value):

    hypos = []

    for id in range(len(input_vars)):
        var = input_vars[id]
        val = input_value[id]

        eql, rhs = [[var], [1]], [float(val)]

        cnames = ['hypo_{0}'.format(id)]
        senses = ['E']
        constr = [eql]

        pb.linear_constraints.add(lin_expr=constr, senses=senses, rhs=rhs, names=cnames)

        # adding a constraint to the list of hypotheses
        hypos.append(tuple([cnames[0], constr, rhs, senses, id]))

    return hypos


def freeze_output_CPLEX(pb, output_vars, output_value):
    # adding indicators for correct and wrong outputs

    pb.variables.add(names=['c_{0}'.format(i) for i in range(len(output_vars))], types='B' * len(output_vars))
    for i in range(len(output_vars)):
        ivar = 'c_{0}'.format(i)
        wrong = ['wc_{0}_{1}'.format(i, j) for j in range(len(output_vars)) if i != j]
        pb.variables.add(names=wrong, types='B' * len(wrong))

        # ivar implies at least one wrong class
        pb.indicator_constraints.add(indvar=ivar, lin_expr=[wrong, [1] * len(wrong)], sense='G', rhs=1)

        for j in range(len(output_vars)):
            if i != j:
                # iv => (o_j - o_i >= 0.0000001)
                iv = 'wc_{0}_{1}'.format(i, j)
                ov, oc = [output_vars[j], output_vars[i]], [1, -1]
                pb.indicator_constraints.add(indvar=iv, lin_expr=[ov, oc], sense='G', rhs=0.0001)

    id_class = output_value.index(1)

    pb.linear_constraints.add(lin_expr=[[['c_{0}'.format(id_class)], [1]]], senses='E', rhs=[1], names=['neg_prediction'])


def contains_variables(variables, all_variables_added):
    return len(variables) > 0 \
           and len(all_variables_added) > 0\
           and len([v for v in variables if v in all_variables_added]) > 0


def define_formula_CPLEX(categorical_ids, A, b):

    n_of_layers = len(A)
    pb = cplex.Cplex()

    all_variables_added = []
    for n_of_layer in range(n_of_layers):
        w_id = 0

        number_of_vars = len(A[n_of_layer][0])
        variables = generate_variables(n_of_layer, categorical_ids, number_of_vars)
        if n_of_layer == 0:
            inputs = variables.copy()

        if not contains_variables(variables, all_variables_added):
            all_variables_added.extend(variables)
            pb.variables.add(names=variables, lb=[0] * len(variables), types=get_type(variables))

        for i in range(len(A[n_of_layer])):
            weights = A[n_of_layer][i].tolist()
            weights.extend([-1, +1])

            rhs = -1 * b[n_of_layer][i]

            support_variables = get_variables_names(w_id, n_of_layer)

            if not support_variables[0] in all_variables_added:
                sup_vars_n = len(support_variables)
                pb.variables.add(names=support_variables, lb=[0] * sup_vars_n, types=["C"] * sup_vars_n)
                all_variables_added.extend(support_variables)

            variables.extend(support_variables)
            var_ids = pb.variables.get_indices(name=variables)
            idx, w = decouple_couple(var_ids, weights)

            pb.linear_constraints.add(lin_expr=[[idx, w]],
                                      rhs=[rhs],
                                      names=["constraint_{0}".format(i)])

            variables = drop_elements_from_list(variables, support_variables)
            set_indicator_constraint(pb, support_variables)
            w_id += 1

    pb.write("encoded_with_cplex_solver.lp")

    return pb, inputs


def delete_hypos_and_output_constraint(pb):

    vars_to_drop = get_vars(pb, ["w"])

    linear_to_drop = [v for v in pb.linear_constraints.get_names() if "hypo" in v]
    indicator_to_drop = [v for v in pb.indicator_constraints.get_names() if "layer" not in v]

    pb.linear_constraints.delete(linear_to_drop)
    pb.indicator_constraints.delete(indicator_to_drop)
    pb.linear_constraints.delete("neg_prediction")

    pb.variables.delete(vars_to_drop)


def compute_minimal_CPLEX(oracle, hypos):
        rhypos = []

        # simple deletion-based linear search
        for i, hypo in enumerate(hypos):
            oracle.linear_constraints.delete(hypo[0])
            oracle.solve()
            if oracle.solution.is_primal_feasible():
                #print([oracle.solution.get_values(v) for v in oracle.variables.get_names()])
                # this hypothesis is needed
                # adding it back to the list
                oracle.linear_constraints.add(lin_expr=hypo[1], senses=hypo[3], rhs=hypo[2], names=[hypo[0]])

                rhypos.append(tuple([hypo[4], hypo[0]]))

        return rhypos

def explain_CPLEX(oracle, hypos):

    oracle.solve()
    if oracle.solution.is_primal_feasible():
        print('no implication!')
        #model = pb.solution
        #print('coex  sample:', [self.coex_model.get_values(i) for i in self.inputs])
        #print('coex  rounded:', [round(self.coex_model.get_values(i)) for i in self.inputs])
        #print('coex classes:', [self.coex_model.get_values(o) for o in self.outputs])
        #print('wrong sample:', [model.get_values(i) for i in self.inputs])
        #print('wrong rounded:', [round(model.get_values(i)) for i in self.inputs])
        #print('wrong classes:', [model.get_values(o) for o in self.outputs])

        sys.exit(1)

    rhypos = compute_minimal_CPLEX(oracle, hypos)

    expl_sz = len(rhypos)
    print('  # hypos left:', expl_sz)

    return rhypos