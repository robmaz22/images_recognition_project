import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

import argparse
from datetime import datetime
import pandas as pd
import os

print(f'Bieżąca wersja TensorFlow: {tf.__version__}')

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, help='Path to images')
parser.add_argument('-e', '--epochs', default=5, type=int, help='number of epochs')
args = vars(parser.parse_args())

learning_rate = 0.001
EPOCHS = args['epochs']
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)
TRAIN_DIR = os.path.join(args['path'], 'train')
VALID_DIR = os.path.join(args['path'], 'valid')

train_datagen = ImageDataGenerator(
    rotation_range=25,
    rescale=1. / 255.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    directory=VALID_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=INPUT_SHAPE, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy'])

model.summary()

dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
filepath = os.path.join('output', 'model_' + dt + '.hdf5')
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', save_best_only=True)

print('[INFO] Trenowanie modelu...')
history = model.fit(
    x=train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // BATCH_SIZE,
    epochs=1,
    callbacks=[checkpoint])

test_gen = ImageDataGenerator(
    rescale=1. / 255.
)

test_generator = test_gen.flow_from_directory(
    directory=os.path.join(args['path'], 'test'),
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

print('[INFO] Wczytywanie wytrenowanego modelu...')
model = load_model(filepath)

y_prob = model.predict(test_generator, workers=1)
y_prob = y_prob.ravel()

y_true = test_generator.classes

predictions = pd.DataFrame({'y_prob': y_prob, 'y_true': y_true}, index=test_generator.filenames)
predictions['y_pred'] = predictions['y_prob'].apply(lambda x: 1 if x > 0.5 else 0)
predictions['is_incorrect'] = (predictions['y_true'] != predictions['y_pred']) * 1
errors = list(predictions[predictions['is_incorrect'] == 1].index)
print(predictions.head())

y_pred = predictions['y_pred'].values

print(f'[INFO] Macierz konfuzji:\n{confusion_matrix(y_true, y_pred)}')
print(f'[INFO] Raport klasyfikacji:\n{classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())}')
print(f'[INFO] Dokładność modelu: {accuracy_score(y_true, y_pred) * 100:.2f}%')

print(f'[INFO] Błędnie sklasyfikowano: {len(errors)}\n[INFO] Nazwy plików:')
for error in errors:
    print(error)
