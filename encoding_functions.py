from pysmt.shortcuts import Symbol, And, Not, is_sat

def define_number_of_outputs(model):
    outpt, inpt = [], []

    for l in ["firstlayer", "secondlayer"]:
        inpt.append(model.get_layer(l).input_shape[1])
        outpt.append(model.get_layer(l).output_shape[1])

    return inpt, outpt

def define_vector_of_weights(model):
    A = []
    b = []
    input_dims, output_dims = define_number_of_outputs(model)
    n_of_layer = 0
    for layer in zip(input_dims, output_dims):
        total_weights = model.layers[n_of_layer].get_weights()
        weights = total_weights[0]
        bias = total_weights[1]
        l = []
        bs = []

        for output_node in range(layer[1]):
            row = []
            for input_node in range(layer[0]):
                 row.append(weights[input_node][output_node])
            l.append(row)
            bs.append(bias[output_node])

        b.append(bs)
        A.append(l)
        n_of_layer = n_of_layer + 1

    print("A: ", len(A))
    for l in A:
        print("layer size: ", len(l))
        for out in l:
            print("len out: ", len(out))

    print(b)