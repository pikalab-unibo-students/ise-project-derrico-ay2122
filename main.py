# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import keras as kr
from keras import layers
from pmlb import fetch_data

def import_dataframe():
    dictionary = {}

    dataframes = ['australian', 'auto', 'backache',
                  'breast_cancer', 'cleve', 'cleveland',
                  'glass', 'glass2', 'heart_statlog',
                  'hepatitis', 'vote', 'spect'
                  ]

    for df_name in dataframes:
        df = fetch_data(df_name)
        dictionary[df_name] = df

    return dictionary

#Procedura di creazione del modello
def build_model(df):

    nodes_number = [10, 15, 20]
    models = []

    for nodes in nodes_number:
        model = kr.Sequential([
            layers.Dense(nodes, activation="relu", input_shape=len(df['target'].unique())),
            layers.Dense(3, activation="relu"),
            layers.Dense(4),
        ])

    #optimizer = tf.keras.optimizers.SGD(0.005)

    #model.compile(loss='sparse_categorical_crossentropy',
                  #optimizer=optimizer,
                  #metrics=['accuracy'])
    return model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dict = import_dataframe()
    print(dict['auto'])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
