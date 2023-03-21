import os
from PIL import Image

cur_path = os.getcwd()
for directory in [name for name in os.listdir(cur_path) if os.path.isdir(cur_path + '\\' + name)]:
    print(cur_path + '\\' + directory)
    for image in os.listdir(cur_path + '\\' + directory):
        if image.endswith('.png'):
            try:
                file = Image.open(cur_path + '\\' + directory + '\\' + image)
                file = file.convert('RGB')
                print(cur_path + '\\' + directory + '\\' + image.split('.')[0] + '.jpg')
                file.save(cur_path + '\\' + directory + '\\' + image.split('.')[0] + '.jpg')
                os.remove(cur_path + '\\' + directory + '\\' + image)
            except:
                os.remove(cur_path + '\\' + directory + '\\' + image)
