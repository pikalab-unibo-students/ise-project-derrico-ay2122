import os
import sys

import numpy as np
import pysmt
from pysmt import *
import cplex
from pysmt.shortcuts import Equals, Symbol, Implies, LE, Int, Real, Times, And, Minus, Solver, GE, Iff, TRUE, FALSE
from pysmt.typing import INT, REAL, BOOL


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
    return vars

def set_type(id, categorical_ids, n_layer):
    return "B" if id in categorical_ids and n_layer == 0 else "C"

def get_type(variables_to_type):
    return [v.split("_")[-1] for v in variables_to_type]

def get_variables_names(id, n_of_layer):
    suffix = '_{0}_layer_{1}_type_C'.format(id, n_of_layer+1)
    return ["X" + suffix, "S" + suffix]

def drop_elements_from_list(initial, to_remove):
    return [v for v in initial if v not in to_remove]

def decouple_couple(var_index, weights):
    return [list(value) for value in zip(*[couple for couple in zip(var_index, weights)])]

def freeze_input_and_output_CPLEX(pb, input_value, output_value):
    hypos = freeze_input_CPLEX(pb, input_value)
    freeze_output_CPLEX(pb, output_value)

    return hypos

def freeze_input_CPLEX(pb, input_value):

    hypos = []

    input_vars = get_vars(pb, ['X', 'layer_0'])

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

def freeze_output_CPLEX(pb, output_value):
    # adding indicators for correct and wrong outputs

    output_vars = get_vars(pb, ['X', 'layer_2'])
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

def generate_variables(layer, categorical_ids, n_vars):
    return ['X_{0}_layer_{1}_type_{2}'.format(i, layer, set_type(i, categorical_ids, layer)) for i in range(n_vars)]

def define_formula_CPLEX(categorical_ids, A, b):

    n_of_layers = len(A)
    pb = cplex.Cplex()

    all_variables_added = []
    for n_of_layer in range(n_of_layers):
        w_id = 0

        number_of_vars = len(A[n_of_layer][0])
        variables = generate_variables(n_of_layer, categorical_ids, number_of_vars)

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

    pb.write("encoded_model.lp")

    return pb

def SMT_indicator_constraints(y, s):
    name = "Z_" + '_'.join(("" + str(y)).split("_")[1:])

    z = Symbol(name)
    constraint_1 = Implies(Iff(z, TRUE()), LE(y, Real(0)))
    constraint_2 = Implies(Iff(z, FALSE()), LE(s, Real(0)))

    constraint_3 = GE(y, Real(0))
    constraint_4 = GE(s, Real(0))

    return And(constraint_1, constraint_2, constraint_3, constraint_4)

def print_formula(fname, formula):
    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(fname, "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("Formula: ", formula)
        sys.stdout = original_stdout  # Reset the standard output to its original value

def cancel_file(fname):
    file = open(fname, "w")
    file.close()

def define_formula_SMT(categorical_ids, A, b, input, output):

    n_of_layers = len(A)

    formula = None
    file_name = "encoded_with_smt_solver"
    solver = Solver(name="z3", logic="QF_UFLRA")
    output_vars = []

    cancel_file(file_name)
    for n_of_layer in range(n_of_layers):

        number_of_vars = len(A[n_of_layer][0])
        variables = generate_variables(n_of_layer, categorical_ids, number_of_vars)
        variables = [Symbol(v, REAL) for v in variables]

        n_vars = len(variables)
        n_outputs = len(A[n_of_layer])

        for id_output in range(n_outputs):
            for id_var in range(n_vars):
                coeff = Real(float(A[n_of_layer][id_output][id_var]))
                new_term = Times(coeff, variables[id_var])
                if id_var == 0:
                    formula = new_term
                else:
                    formula += new_term

            support_variables = get_variables_names(id_output, n_of_layer)
            y = Symbol(support_variables[0], REAL)
            s = Symbol(support_variables[1], REAL)

            if n_of_layer == 1:
                output_vars.append(y)

            constant_term = Real(float(b[n_of_layer][id_output]))
            formula = Equals(Minus(formula, constant_term),
                             Minus(y, s))

            print_formula(file_name, formula)
            solver.add_assertion(formula)

            ic = SMT_indicator_constraints(y, s)
            solver.add_assertion(ic)
            print_formula(file_name, ic)

        if n_of_layer == 0:
            hypos = [Equals(variables[i], Real(float(input[i]))) for i in range(len(variables))]
            for c in hypos:
                print_formula(file_name, c)

    freeze_output_SMT(solver, output_vars, output)

    return solver, hypos

def freeze_output_SMT(solver, output_vars, output_value):

    for i in range(len(output_vars)):
        wrong = [Symbol('wc_{0}_{1}'.format(i, j)) for j in range(len(output_vars)) if i != j]

        c = wrong[0]
        if len(wrong) > 1:
            for k in range(1, len(wrong)):
                c += wrong[k]

        if i == output_value.index(1):
            ivar = TRUE()
        else:
            ivar = Symbol('c_{0}'.format(i))

        assertion1 = Implies(Iff(ivar, TRUE()), Iff(c, TRUE()))
        print_formula("encoded_with_smt_solver", assertion1)

        # ivar implies at least one wrong class
        solver.add_assertion(assertion1)

        for j in range(len(output_vars)):
            if i != j:
                # iv => (o_j - o_i >= 0.0000001)
                iv = Symbol('wc_{0}_{1}'.format(i, j))
                right = output_vars[j] - output_vars[i]
                rhs = Real(0.0001)

                assertion2 = Implies(Iff(iv, TRUE()), GE(right, rhs))
                print_formula("encoded_with_smt_solver", assertion2)
                solver.add_assertion(assertion2)

def compute_minimal_CPLEX(oracle, hypos):
        rhypos = []

        # simple deletion-based linear search
        for i, hypo in enumerate(hypos):
            oracle.linear_constraints.delete(hypo[0])
            oracle.solve()
            if oracle.solution.is_primal_feasible():
                # this hypothesis is needed
                # adding it back to the list
                oracle.linear_constraints.add(lin_expr=hypo[1], senses=hypo[3], rhs=hypo[2], names=[hypo[0]])

                rhypos.append(tuple([hypo[4], hypo[0]]))

        return rhypos

def explain_CPLEX(oracle, hypos):

    oracle.solve()
    if oracle.solution.is_primal_feasible():
        print('  no implication!')
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

def compute_minimal_SMT(solver, hypos):
    rhypos = []

    # simple deletion-based linear search
    for hypo in hypos:
        s = copy_solver(solver)
        hypos_temp = hypos.copy()
        hypos_temp.remove(hypo)

        for h in hypos_temp:
            s.add_assertion(h)

        if s.solve():
            # this hypothesis is needed
            # adding it back to the list
            rhypos.append(hypo)

    return rhypos

def copy_solver(solver):
    s = Solver(name="z3", logic="QF_UFLRA")
    s.add_assertions(solver.assertions)
    return s

def explain_SMT(solver, hypos):

    s = copy_solver(solver)
    for h in hypos:
        s.add_assertion(h)

    if s.solve():
        print('  no implication!')
        print(s.get_model())
        sys.exit(1)

    rhypos = compute_minimal_SMT(solver, hypos)

    return rhypos

def encoding_model(model, df_name, solver):
    assert solver in ["smt", "cplex"]

    N_OF_LAYERS = 2
    A, b = get_weights_and_bias(model, N_OF_LAYERS)
    categorical_ids = read_categorical_indexes(df_name)

    if solver == "cplex":
        pb = define_formula_CPLEX(categorical_ids, A, b)
    elif solver == "smt":
        None
        #solver = define_formula_SMT(categorical_ids, A, b, input_values, output_values)
        #rhypos = explain_SMT(solver, hypos)

    return pb

def explaining_procedures(pb, solver, input_values, output_values):

    if solver == "cplex":
        hypos = freeze_input_and_output_CPLEX(pb, input_values, output_values)
        rhypos = explain_CPLEX(solver, hypos)
    elif solver == "smt":
        None
        #solver = define_formula_SMT(categorical_ids, A, b, input_values, output_values)
        #rhypos = explain_SMT(solver, hypos)

    rhypos = None

    print("LEN RHYPOS: ", len(rhypos))
    return len(rhypos)