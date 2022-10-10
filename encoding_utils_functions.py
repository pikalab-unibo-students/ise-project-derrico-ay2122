def set_type(id, categorical_ids, n_layer):
    return "B" if id in categorical_ids and n_layer == 0 else "C"

def generate_variables(layer, categorical_ids, n_vars):
    return ['X_{0}_layer_{1}_type_{2}'.format(i, layer, set_type(i, categorical_ids, layer)) for i in range(n_vars)]

def get_variables_names(id, n_of_layer):
    suffix = '_{0}_layer_{1}_type_C'.format(id, n_of_layer+1)
    return ["X" + suffix, "S" + suffix]