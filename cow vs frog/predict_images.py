from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, help='Path to file/files')
parser.add_argument('-m', '--model', required=True, help='Path to model')
args = vars(parser.parse_args())

file_path = args['path']

print('[INFO] Wczytywanie zdjęcia...')

def load(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (150, 150))
    image = image.astype('float') / 255.
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

image = load(file_path)

print('[INFO] Wczytanie modelu...')
model = load_model(args['model'])

print('[INFO] Analiza zdjęcia...')

class_ = model.predict_classes(image)

if 0 in class_:
    class_ = 'Cow'
else:
    class_ = 'Frog'

img = cv2.imread(filename=file_path)

cv2.putText(img=img,
            text=f'PREDICTION: {class_}',
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            org=(30,30),
            fontScale=1.2,
            color=(0,0,255),
            thickness=2)

cv2.imshow('image', img)
while True:
    cv2.waitKey(1)
    if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) <1:
        break
cv2.destroyAllWindows()
