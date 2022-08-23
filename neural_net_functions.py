import keras as kr
from keras import layers

#Procedura di creazione del modello
def build_model(df, df_name):

    nodes_number = [10, 15, 20]
    models = []

    input_shape = len(df.columns) - 1
    classes = len(df['target'].unique())

    dataset_and_models = {}

    for nodes in nodes_number:

        model = kr.Sequential([
            layers.Dense(nodes, activation="relu", input_shape=[input_shape]),
            layers.Dense(classes, activation="softmax")
        ])
        #model.compile()
        models.append(model)

    dataset_and_models[df_name] = models
    return dataset_and_models