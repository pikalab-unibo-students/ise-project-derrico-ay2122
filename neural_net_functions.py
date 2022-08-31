import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots as plots
#from tensorflow.python.client import device_lib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

import seaborn as sns

def dataset_preprocessing(df):

    df_new = df.copy()
    categorical_features = [column for column in df.columns.values if "_categorical" in column]
    categorical_names = {}

    print(df.columns)
    if categorical_features != []:
        for feature in categorical_features:
            print("feature", feature)
            print(df.loc[:, feature])
            le = LabelEncoder()
            le.fit(df.loc[:, feature])
            categorical_names[feature] = le.classes_
            df_new.loc[:, feature] = le.transform(df.loc[:, feature])

        temp = pd.get_dummies(df_new.loc[:, categorical_features], columns=categorical_features)
        df_new = df_new.loc[:, [v for v in df.columns if v not in categorical_features]].join(temp)
        print(df_new.columns)

    # target as categorical
    le_target = LabelEncoder()
    df_new.loc[:, "target"] = le_target.fit_transform(df.loc[:, "target"])

    #print(df_new['target'].unique())

    return df_new

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
    plt.ylabel('Loss')
    plt.show()

def cross_fold(inputs, targets, model):

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=5, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        model.fit(inputs, targets, epochs=70, shuffle=True, callbacks=[early_stop])
        #plot_accuracy(history)

        # Generate generalization metrics
        scores = model.evaluate(inputs[test], targets[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

    print("Average accuracy: ", np.mean(acc_per_fold))
    print("Average loss: ", np.mean(loss_per_fold))

def fit_model(df, model):

    x, y = create_datasets(df)

    cross_fold(x, y, model)

    y_pred = model.predict(x)

    y_pred = y_pred.argmax(axis=-1)

    print('Accuratezza:', accuracy_score(y, y_pred) * 100, '%')

    plt.figure(figsize=(16, 16))
    cm = confusion_matrix(y, y_pred)
    f = sns.heatmap(cm, annot=True)
    plt.show()

    #print("Accuracy: ", len(y_pred[y_pred != y_test]) / len(y_test))

    return model

#Procedura di creazione del modello
def build_model(df, df_name):

    nodes_number = [20]
    models = []

    df = dataset_preprocessing(df)

    input_shape = len(df.columns) - 1
    classes = len(df['target'].unique())

    dataset_and_models = {}

    for nodes in nodes_number:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nodes, activation='relu',
                                  input_shape=[input_shape],
                                  kernel_regularizer=tf.keras.regularizers.L2(0.0001)),
            tf.keras.layers.Dense(classes, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])

        fit_model(df, model)
        models.append(model)

    dataset_and_models[df_name] = models
    return dataset_and_models