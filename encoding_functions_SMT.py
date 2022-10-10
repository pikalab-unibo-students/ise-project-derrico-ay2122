import sys

from pysmt.shortcuts import Symbol, Implies, Iff, TRUE, LE, Real, FALSE, GE, And, Solver, Equals, Minus, Times, Int, GT, \
    Or, Plus, Not, ExactlyOne
from pysmt.typing import REAL, INT, BOOL

from encoding_utils_functions import generate_variables, get_variables_names


def SMT_indicator_constraints(y, s):
    name = "Z_" + '_'.join(("" + str(y)).split("_")[1:])

    z = Symbol(name, BOOL)

    constraint = []
    constraint.append(Iff(Iff(z, TRUE()), LE(y, Real(0))))
    constraint.append(Iff(Iff(z, FALSE()), LE(s, Real(0))))

    constraint.append(GE(y, Real(0)))
    constraint.append(GE(s, Real(0)))

    return constraint


def print_solver_assertion(fname, solver):
    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open(fname, "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        for a in solver.assertions:
            print("Assertion: ", a)

        sys.stdout = original_stdout  # Reset the standard output to its original value


def cancel_file(fname):
    file = open(fname, "w")
    file.close()


def copy_solver(solver):
    s = Solver(name="z3")
    s.add_assertions(solver.assertions)
    return s


def define_formula_SMT(categorical_ids, A, b):

    n_of_layers = len(A)

    formula = None
    solver = Solver(name="z3")
    all_vars = {}

    for n_of_layer in range(n_of_layers):

        number_of_vars = len(A[n_of_layer][0])
        variables_names = generate_variables(n_of_layer, categorical_ids, number_of_vars)
        if all_vars == [] or not all(item in all_vars.keys() for item in variables_names):
            to_add = {v_name: Symbol(v_name, REAL) for v_name in variables_names}
            all_vars.update(to_add)
            variables = list(to_add.values())
        else:
            variables = [v[1] for v in all_vars.items() if v[0] in variables_names]

        categorical_boundaries = [ExactlyOne(Equals(v, Real(0)), Equals(v, Real(1))) for v in
                       [all_vars[k] for k in all_vars.keys() if "_type_B" in k and all_vars[k] in variables]]

        solver.add_assertions(categorical_boundaries)

        n_vars = len(variables)
        n_outputs = len(A[n_of_layer])

        if n_of_layer == 0:
            inputs = variables.copy()

        for id_output in range(n_outputs):
            for id_var in range(n_vars):
                coeff = Real(float(A[n_of_layer][id_output][id_var]))
                new_term = Times(coeff, variables[id_var])
                #print("Addendo: ", coeff, "*", variables[id_var])
                if id_var == 0:
                    formula = new_term
                else:
                    formula += new_term

            support_variables = get_variables_names(id_output, n_of_layer)
            y = Symbol(support_variables[0], REAL)
            s = Symbol(support_variables[1], REAL)
            all_vars[support_variables[0]] = y
            all_vars[support_variables[1]] = s

            constant_term = Real(float(b[n_of_layer][id_output]))
            formula = Equals(Plus(formula, constant_term),
                             Minus(y, s))

            solver.add_assertion(formula)

            ic = SMT_indicator_constraints(y, s)
            for constraint in ic:
                solver.add_assertion(constraint)

    return solver, inputs


def freeze_input_and_output_SMT(pb, inputs, outputs):

    input_vars, input_values = inputs
    output_vars, output_values = outputs

    hypos = get_hypos_SMT(input_vars, input_values)
    freeze_output_SMT(pb, output_vars, output_values)

    return hypos


def get_hypos_SMT(input_vars, input_value):
    return [Equals(input_vars[i], Real(float(input_value[i]))) for i in range(len(input_vars))]


def set_bound(var):
    return And(LE(var, Int(1)), GE(var, Int(0)))


def freeze_output_SMT(solver, output_vars, output_value):

    output_vars = [Symbol(v, typename=REAL) for v in output_vars]
    output_id = output_value.index(1)

    disj = []
    for i in range(len(output_value)):
        if i != output_id:
            disj.append(GT(output_vars[i], output_vars[output_id]))

    solver.add_assertion(Or(disj))


def explain_SMT(solver, hypos):

    s = copy_solver(solver)

    if s.solve(hypos):
        print('no implication!')
        print(s.get_model())
        sys.exit(1)

    fname = "encoded_with_smt_solver"
    cancel_file(fname)
    print_solver_assertion(fname, s)

    rhypos = compute_minimal_SMT(copy_solver(solver), hypos)

    return rhypos


def compute_minimal_SMT(solver, hypos):

    rhypos = hypos.copy()

    i = 0

    while i < len(rhypos):
        s = copy_solver(solver)
        to_test = rhypos[:i] + rhypos[(i + 1):]

        if s.solve(to_test):
            print("solution find")
            i += 1
        else:
            print("solution not find")
            rhypos = to_test

   # # simple deletion-based linear search
   #  for hypo in hypos:
   #      s = copy_solver(solver)
   #      hypos_temp = hypos.copy()
   #      hypos_temp.remove(hypo)
   #
   #      for h in hypos_temp:
   #          s.add_assertion(h)
   #
   #      if not s.solve():
   #          # this hypothesis is needed
   #          # adding it back to the list
   #          rhypos.append(hypo)

    return rhypos