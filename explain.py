import time

from encoding_functions_CPLEX import freeze_input_and_output_CPLEX, explain_CPLEX, delete_hypos_and_output_constraint, \
    get_weights_and_bias, read_categorical_indexes, define_formula_CPLEX
from encoding_functions_SMT import define_formula_SMT, explain_SMT, freeze_input_and_output_SMT, copy_solver

def encoding_model(model, df_name, solver):
    assert solver in ["smt", "cplex"]

    N_OF_LAYERS = 2
    A, b = get_weights_and_bias(model, N_OF_LAYERS)
    categorical_ids = read_categorical_indexes(df_name)

    if solver == "cplex":
        pb, inputs = define_formula_CPLEX(categorical_ids, A, b)
    elif solver == "smt":
        pb, inputs = define_formula_SMT(categorical_ids, A, b)

    return pb, inputs


def explaining_procedures(pb, solver_name, inputs, outputs):
    rhypos = []

    if solver_name == "cplex":
        hypos = freeze_input_and_output_CPLEX(pb, inputs, outputs)

        start_time = time.time()
        rhypos = explain_CPLEX(pb, hypos)
        end_time = time.time() - start_time

        delete_hypos_and_output_constraint(pb)
    elif solver_name == "smt":
        s = copy_solver(pb)
        hypos = freeze_input_and_output_SMT(s, inputs, outputs)
        start_time = time.time()
        rhypos = explain_SMT(s, hypos)
        end_time = time.time() - start_time

    print("LEN RHYPOS: ", len(rhypos))
    return len(rhypos), end_time