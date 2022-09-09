import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_docs.plots as plots
from keras import Sequential
from keras.layers import Dense
from keras.models import clone_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import seaborn as sns

def dataset_preprocessing(df, df_name):

    df_new = df.copy()
    categorical_features = [column for column in df.columns.values if ("_categorical" or "_boolean") in column]
    categorical_names = {}

    if categorical_features != []:
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(df.loc[:, feature])
            categorical_names[feature] = le.classes_
            df_new.loc[:, feature] = le.transform(df.loc[:, feature])

        temp = pd.get_dummies(df_new.loc[:, categorical_features], columns=categorical_features)
        df_new = df_new.loc[:, [v for v in df.columns if v not in categorical_features]].join(temp)

    # target as categorical
    le_target = LabelEncoder()
    df_new.loc[:, "target"] = le_target.fit_transform(df.loc[:, "target"])

    print_categorical_indexes(df_new, df_name)

    return df_new

def print_categorical_indexes(df, df_name):
    ids = [df.columns.get_loc(col) for col in df.columns if "_" in col]
    with open(".\datasets_boolean_index\\" + df_name + "\\" + df_name + '_categorical_indexes.txt', 'w') as f:
        for id in ids:
            f.write(str(id) + "\n")

def create_datasets(df):

    columns = [c for c in df.columns if c != 'target']
    x, y = df.loc[:, columns], df.loc[:, 'target']
    # Applica nomralizzazione con StandardScaler di scikit-learn
    x = StandardScaler().fit_transform(x)

    return x, y

def plot_accuracy(history):
    plotter = plots.HistoryPlotter(smoothing_std=0)

    plotter.plot({'Basic': history}, metric="accuracy")
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.show()

    plotter.plot({'Basic': history}, metric="loss")
    plotter.plot({'Basic': history}, metric="loss")
    plt.ylabel('Loss')
    plt.show()

def model_compiling(model):
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

def get_model(params):

    nodes = params[1]
    input_shape = params[0]
    classes_number = params[2]

    model = Sequential([
        Dense(nodes, activation='relu', name="firstlayer",
              input_shape=[input_shape], bias_initializer=tf.initializers.Constant(0.1)),
        Dense(classes_number, name="secondlayer", activation='softmax')
    ])

    #model_compiling(model)

    return model

def cross_fold(inputs, targets, model):

    # Define the K-fold Cross Validator
    stratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1

    for train, test in stratifiedKFold.split(inputs, targets):

        crossed_model = clone_model(model)
        model_compiling(crossed_model)

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        crossed_model.fit(inputs, targets, batch_size=4, epochs=50, shuffle=True, callbacks=[early_stop])

        # Generate generalization metrics
        scores = crossed_model.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {crossed_model.metrics_names[0]} of {scores[0]}; {crossed_model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    print("Average accuracy: ", np.mean(acc_per_fold))
    print("Average loss: ", np.mean(loss_per_fold))

def fit_model(df, parameters, df_name):

    x, y = create_datasets(df)

    model = get_model(parameters)
    cross_fold(x, y, model)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)
    #
    # model.fit(x_train, y_train, batch_size=4, epochs=50, shuffle=True)
    #
    # y_pred = model.predict(x_test)
    # y_pred = y_pred.argmax(axis=-1)
    #
    # print('Accuratezza:', accuracy_score(y_test, y_pred) * 100, '%')
    #
    # plt.figure(figsize=(16, 16))
    # cm = confusion_matrix(y_test, y_pred)
    # f = sns.heatmap(cm, annot=True)
    # plt.show()

    return model

#Procedura di creazione del modello
def build_model(df, df_name):

    nodes_number = [20]
    models = []

    df = dataset_preprocessing(df, df_name)

    input_shape = len(df.columns) - 1
    classes = len(df['target'].unique())

    dataset_and_models = {}

    for nodes in nodes_number:

        model = fit_model(df, (input_shape, nodes, classes), df_name)
        models.append(model)

    dataset_and_models[df_name] = models
    return dataset_and_models