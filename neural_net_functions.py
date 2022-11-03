import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_docs.plots as plots
from keras import Sequential
from keras.layers import Dense
from keras.models import clone_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def mnist_preprocessing(df):
    le_target = LabelEncoder()
    df.loc[:, "target"] = le_target.fit_transform(df.loc[:, "target"])
    max_value = 255

    target, df = df['target'], df.drop(columns='target')

    for j in range(df.shape[0]):
        for i in range(df.shape[1]):
            v = df.iloc[j, i]
            diff_1 = max_value - v
            df.iloc[j, i] = 0 if diff_1 > int(max_value / 2) else 1

    categorical_ids = [v for v in range(len(df.columns))]
    print_categorical_indexes(categorical_ids, "mnist")

    df['target'] = target

    return df


def dataset_preprocessing(df, df_name=None):

    if df_name == "mnist":
        return mnist_preprocessing(df)
    else:
        df_new = df.copy()
        categorical_features = [column for column in df.columns.values if "_categorical" in column]
        categorical_names = {}

        there_are_categorical_features = categorical_features != []

        if there_are_categorical_features:
            for feature in categorical_features:
                le = LabelEncoder()
                le.fit(df.loc[:, feature])
                categorical_names[feature] = le.classes_
                df_new.loc[:, feature] = le.transform(df.loc[:, feature])

            #temp = pd.get_dummies(df_new.loc[:, categorical_features], columns=categorical_features)
            #df_new = df_new.loc[:, [v for v in df.columns if v not in categorical_features]].join(temp)

        # target as categorical
        le_target = LabelEncoder()
        df_new.loc[:, "target"] = le_target.fit_transform(df.loc[:, "target"])

        for column in df.columns.values:
            if "_boolean" in column:
                there_are_boolean = True
                categorical_features.append(column)

        if df_name is not None:
            ite = filter(lambda x: [v for v in categorical_features if v in x], df_new.columns)
            ids = [df_new.drop(columns="target").columns.get_loc(c) for c in list(ite)]

            attrs = []
            for i in ids:
                attrs.append([i, df_new.iloc[:, i].min(), df_new.iloc[:, i].max()])

            if there_are_categorical_features or there_are_boolean:
                print_categorical_indexes(attrs, df_name)

        return df_new


def print_categorical_indexes(attrs, df_name):
    with open(".\datasets_categorical_index\\" + df_name + "\\" + df_name + '_categorical_indexes.txt', 'w') as f:
        for v in attrs:
            f.write(str(int(v[0])) + "_" + str(int(v[1])) + "_" + str(int(v[2])) + "\n")


def create_datasets(df):

    columns = [c for c in df.columns if c != 'target']
    x, y = df.loc[:, columns], df.loc[:, 'target']
    print("Colonne: ", x.columns)
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
                  run_eagerly=True,
                  optimizer="adam",
                  metrics=['accuracy'])


def get_model(params):

    nodes = params[1]
    input_shape = params[0]
    classes_number = params[2]

    model = Sequential([
        Dense(nodes, activation='relu', name="firstlayer", input_dim=input_shape),
        Dense(classes_number, name="secondlayer", activation='softmax')
    ])

    #model_compiling(model)

    return model


def cross_fold(inputs, targets, model):

    # Define the K-fold Cross Validator
    stratifiedKFold = StratifiedKFold(n_splits=3, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1

    for train, test in stratifiedKFold.split(inputs, targets):

        crossed_model = clone_model(model)
        model_compiling(crossed_model)

        print(train.shape)
        print(test.shape)

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        crossed_model.fit(inputs[train], targets[train], batch_size=4, epochs=180, shuffle=True, callbacks=[early_stop])

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
    #cross_fold(x, y, model)
    model_compiling(model)
    model.fit(x, y, batch_size=4, epochs=10, shuffle=True)

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
def build_model(df, hidden_layer_nodes, df_name):

    input_shape = len(df.columns) - 1
    classes = len(df['target'].unique())

    model = fit_model(df, (input_shape, hidden_layer_nodes, classes), df_name)

    return model