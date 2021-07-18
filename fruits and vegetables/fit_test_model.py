# Link do zbioru danych: https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition

import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf
import os
import pandas as pd
import argparse
from datetime import datetime
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_hist(history, output):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Dokładność modelu', 'Funkcja straty modelu'])

    fig.add_trace(
        go.Scatter(x=hist['epoch'],
                   y=hist['accuracy'] * 100,
                   name='Zbiór treningowy',
                   mode='markers+lines'),
        row=1,
        col=1)

    fig.add_trace(
        go.Scatter(x=hist['epoch'],
                   y=hist['val_accuracy'] * 100,
                   name='Zbiór walidacyjny',
                   mode='markers+lines'),
        row=1,
        col=1)

    fig.add_trace(
        go.Scatter(x=hist['epoch'],
                   y=hist['loss'],
                   mode='markers+lines',
                   name='Zbiór treningowy', ),

        row=1,
        col=2)
    fig.add_trace(
        go.Scatter(x=hist['epoch'],
                   y=hist['val_loss'],
                   mode='markers+lines',
                   name='Zbiór walidacyjny', ),
        row=1,
        col=2)

    fig.update_yaxes(title_text='Dokładność [%]', row=1, col=1)
    fig.update_yaxes(title_text='Wartość funkcji straty', row=1, col=2)
    fig.update_xaxes(title_text='Epoki', row=1, col=1)
    fig.update_xaxes(title_text='Epoki', row=1, col=2)

    fig.update_layout(title='<b>Porównanie statystyk modelu<b>')
    fig.show()

    fig.write_html(output)


def build_model():
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(225, 225, 3)))
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', ))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dense(units=36, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='SparseCategoricalCrossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', required=True, help='Path to datasets')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
    args = vars(parser.parse_args())

    return args


def run():
    print(f'Bieżąca wersja TensorFlow: {tf.__version__}')
    print('[INFO] Wczytywanie danych do trenowania...')
    args = get_args()

    train_dir = f"{args['path']}/train"
    valid_dir = f"{args['path']}/validation"
    test_dir = f"{args['path']}/test"
    batch_size = 32
    img_size = (225, 225)
    epochs = args['epochs']

    print('* Zbiór testowy:')
    train_set = image_dataset_from_directory(train_dir,
                                             seed=123,
                                             image_size=img_size,
                                             batch_size=batch_size)

    print('* Zbiór walidacyjny:')
    valid_set = image_dataset_from_directory(valid_dir,
                                             seed=123,
                                             image_size=img_size,
                                             batch_size=batch_size)

    print('Test set:')
    test_set = image_dataset_from_directory(test_dir,
                                            seed=123,
                                            image_size=img_size,
                                            batch_size=batch_size)

    print('[INFO] Tworzenie sieci neuronowej...')
    model = build_model()

    print('[INFO] Trenowanie modelu...')
    os.mkdir('output')
    dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
    es = EarlyStopping(monitor='val_loss', patience=3)
    filepath_model = os.path.join('output', 'model_' + dt + '.hdf5')
    mc = ModelCheckpoint(filepath=filepath_model,
                         save_weights_only=True,
                         monitor='val_accuracy',
                         mode='max',
                         save_best_only=True)

    history = model.fit(x=train_set,
                        epochs=epochs,
                        validation_data=valid_set,
                        callbacks=[es, mc])

    print('[INFO] Eksport statystyk do pliku html...')
    filepath_stat = os.path.join('output', 'model_' + dt + '.html')
    plot_hist(history, filepath_stat)

    print('[INFO] Wczytywanie wytrenowanego modelu...')
    model.load_weights(filepath_model)
    accuracy = model.evaluate(test_set)
    print(f'[INFO] Ocena uzyskanego modelu: {round(accuracy[1] * 100, 2)}%')


if __name__ == '__main__':
    run()
