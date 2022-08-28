import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def fit_model(df, model):
    columns = [c for c in df.columns if c != 'target']
    X, y = df.loc[:, columns], df.loc[:, 'target']

    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Applica nomralizzazione con StandardScaler di scikit-learn
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model

#Procedura di creazione del modello
def build_model(df, df_name):

    nodes_number = [10, 15, 20]
    models = []

    input_shape = len(df.columns) - 1
    classes = len(df['target'].unique())

    dataset_and_models = {}

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