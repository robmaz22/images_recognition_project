import shutil
import os
from random import shuffle
import argparse


parser = argparse.ArgumentParser(description='Preparing data for neural network')
parser.add_argument('-p', "--path", help="Path to images directory")
args = vars(parser.parse_args())

file_path = args['path']

classes = ['frog', 'cow']

print(f'[INFO] Zmiana nazw plików...')
for class_ in classes:
    i = 1
    for name in os.listdir(f'{file_path}/{class_}'):
        new_name = f'{file_path}/{class_}/{i:06d}.jpg'
        os.rename(f'{file_path}/{class_}/{name}', new_name)
        i += 1

dirs = ['train', 'test', 'valid']
base_dir = f'{file_path}/data'

print(f'[INFO] Tworzenie folderów {", ".join(dirs)} dla klas {", ".join(classes)}...')
for dir in dirs:
    for class_ in classes:
        os.makedirs(os.path.join(base_dir, dir, class_))

number_of_pictures = len(os.listdir(f'{file_path}/{classes[0]}'))

train_size = int(number_of_pictures * 0.7)
test_size = int(number_of_pictures * 0.2)
valid_size = int(number_of_pictures * 0.1)

print(f'[INFO] Podział danych ({number_of_pictures}) dla jednej klasy: ')
print(f'Liczba danych do trenowania: {train_size}')
print(f'Liczba danych do testu: {test_size}')
print(f'Liczba danych do walidacji: {valid_size}')

frog_fnames = [fname for fname in os.listdir(f'{file_path}/frog')]
cow_fnames = [fname for fname in os.listdir(f'{file_path}/cow')]

shuffle(frog_fnames)
shuffle(cow_fnames)

lists = {'frog': frog_fnames,
         'cow': cow_fnames}

print('[INFO] Kopiowanie plików do odpowiednich folderów...')
for class_ in classes:
    idx = 0
    while idx < number_of_pictures:

        scr = os.path.join(file_path, class_, lists[class_][idx])
        if idx < train_size:
            dst = os.path.join(file_path, 'data', 'train', class_, lists[class_][idx])
        elif idx > train_size and idx <= (train_size + test_size):
            dst = os.path.join(file_path, 'data', 'test', class_, lists[class_][idx])
        else:
            dst = os.path.join(file_path, 'data', 'valid', class_, lists[class_][idx])
        shutil.copy(scr, dst)
        idx += 1

print('[INFO] Zakończono preprocesing danych.')
