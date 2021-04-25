import shutil
import os
from random import shuffle


filename = 'images.zip'
shutil.unpack_archive(filename)

classes = ['frog', 'cow']

for class_ in classes:
    i = 1
    for name in os.listdir(f'images/{class_}'):
        print(name)
        os.rename(f'images/{class_}/{name}', f'images/{class_}/{i:04d}.jpg')
        i += 1

dirs = ['train', 'test', 'valid']
base_dir = 'data'

for dir in dirs:
    for class_ in classes:
        os.makedirs(os.path.join(base_dir, dir, class_))

number_of_pictures = len(os.listdir('images/cow'))

train_size = int(number_of_pictures * 0.7)
test_size = int(number_of_pictures * 0.2)
valid_size = int(number_of_pictures * 0.1)

print(f'Liczba danych do trenowania: {train_size}')
print(f'Liczba danych do testu: {test_size}')
print(f'Liczba danych do walidacji: {valid_size}')

frog_fnames = [fname for fname in os.listdir('images/frog')]
cow_fnames = [fname for fname in os.listdir('images/cow')]

shuffle(frog_fnames)
shuffle(cow_fnames)

lists = {'frog':frog_fnames,
         'cow':cow_fnames}

for class_ in classes:
    idx = 0
    while idx < 400:

      scr = os.path.join('images', class_, lists[class_][idx])
      if idx < train_size:
        dst = os.path.join('data', 'train', class_, lists[class_][idx])
      elif idx > train_size and idx =< (train_size + test_size):
        dst = os.path.join('data', 'test', class_, lists[class_][idx])
      else:
        dst = os.path.join('data', 'valid', class_, lists[class_][idx])
      shutil.copy(scr, dst)
      idx += 1
