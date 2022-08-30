import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots as plots
#from tensorflow.python.client import device_lib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split
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

    #print(df.groupby("target").count())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, stratify=y)

    # Applica nomralizzazione con StandardScaler di scikit-learn
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

def plot_accuracy(history):
    plotter = plots.HistoryPlotter(smoothing_std=0)

    plotter.plot({'Basic': history}, metric="accuracy")
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.show()

    plotter.plot({'Basic': history}, metric="loss")
    plt.ylabel('Loss')
    plt.show()

def fit_model(df, model):

    X_train, X_test, y_train, y_test = create_datasets(df)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), shuffle=True, callbacks=[early_stop])
    plot_accuracy(history)

    y_pred = model.predict(X_test)

    y_pred = y_pred.argmax(axis=-1)

    print('Accuratezza:', accuracy_score(y_test, y_pred) * 100, '%')

    plt.figure(figsize=(16, 16))
    cm = confusion_matrix(y_test, y_pred)
    f = sns.heatmap(cm, annot=True)
    plt.show()

    #print("Accuracy: ", len(y_pred[y_pred != y_test]) / len(y_test))

    return model

#Procedura di creazione del modello
def build_model(df, df_name):

    nodes_number = [10, 15, 20]
    models = []

    df = dataset_preprocessing(df)

    input_shape = len(df.columns) - 1
    classes = len(df['target'].unique())

    dataset_and_models = {}

    for nodes in nodes_number:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nodes, activation='relu', input_shape=[input_shape]),
            tf.keras.layers.Dense(classes, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.SGD(0.005)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        fit_model(df, model)
        models.append(model)

    dataset_and_models[df_name] = models
    return dataset_and_models