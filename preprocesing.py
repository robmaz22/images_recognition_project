import shutil
import os


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




