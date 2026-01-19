import os
import shutil

data_source = 'resources/UTKFace'
data_dir_train = 'resources/data/train'
data_dir_val = 'resources/data/val'
data_dir_test = 'resources/data/test'

data_dirs = [data_dir_train, data_dir_val, data_dir_test]

for data_dir in data_dirs:
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

for data_dir in data_dirs:
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

def copy_data(dict: dict, key, train_amount, val_amount, test_amount):
    imgs_list = dict.get(key)
    for x in range(train_amount):
        shutil.copy(imgs_list[x], data_dir_train)
    for x in range(train_amount, (train_amount+val_amount)):
        shutil.copy(imgs_list[x], data_dir_val)
    for x in range((train_amount+val_amount), (train_amount+val_amount+test_amount)):
        shutil.copy(imgs_list[x], data_dir_test)

def augment(dict, key, number_of_pictures):
    return

data_dict = {}

for filename in os.listdir(data_source):
    if filename.endswith(".jpg"):
        age = int(filename.split('_')[0])
        img_path = os.path.join(data_source, filename)
        list_of_imgs = data_dict.get(age, [])
        list_of_imgs.append(img_path)
        data_dict.update({age: list_of_imgs})


# Treningowy 70%
# Walidacyjny 20%
# Testowy 10%

# Posortowane według klucza
# data_dict = {k: v for k, v in sorted(data_dict.items(), key=lambda item: item[0])}

for age in data_dict:
    number_of_pictures = len(data_dict.get(age))
    if number_of_pictures < 10:
        # augmenting needed
        continue
    train_amount = int(number_of_pictures*0.7)
    val_amount = int(number_of_pictures*0.2)
    test_amount = int(number_of_pictures*0.1)
    sum = train_amount+val_amount+test_amount
    if sum < number_of_pictures:
        test_amount += number_of_pictures-sum
    
    print(f'Age: {age}, total amount: {number_of_pictures}')
    print(f'Train amount: {train_amount}')
    print(f'Validation amount: {val_amount}')
    print(f'Test amount: {test_amount}', end="\n\n")
    copy_data(data_dict, age, train_amount, val_amount, test_amount)