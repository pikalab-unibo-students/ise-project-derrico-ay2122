import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow_docs as tfdocs
#import tensorflow_docs.modeling
import tensorflow_docs.plots as tfdocs
#from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_datasets(df):

    df[df['target'] < 0] = len(df['target'].unique()) - 1
    columns = [c for c in df.columns if c != 'target']
    X, y = df.loc[:, columns], df.loc[:, 'target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

    # Applica nomralizzazione con StandardScaler di scikit-learn
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def plot_accuracy(history):
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=0)

    plotter.plot({'Basic': history}, metric="accuracy")
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.show()

    plotter.plot({'Basic': history}, metric="loss")
    plt.ylabel('Loss')
    plt.show()

def fit_model(df, model):

    X_train, X_test, y_train, y_test = create_datasets(df)

    history = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred = y_pred.argmax(axis=-1)

    print('Accuratezza sul test set:', accuracy_score(y_test, y_pred) * 100, '%')

    plot_confusion_matrix(y_test, y_pred,
                          classes=df['targets'].unique())

    #print("Accuracy: ", len(y_pred[y_pred != y_test]) / len(y_test))

    return model

#Procedura di creazione del modello
def build_model(df, df_name):

    nodes_number = [10, 15, 20]
    models = []

    input_shape = len(df.columns) - 1
    classes = len(df['target'].unique())

    dataset_and_models = {}

    print(df_name)

    for nodes in nodes_number:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(nodes, activation='relu', input_shape=[input_shape]),
            tf.keras.layers.Dense(classes, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        fit_model(df, model)
        models.append(model)

    dataset_and_models[df_name] = models
    return dataset_and_models