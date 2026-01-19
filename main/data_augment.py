import os
import image_processor
import numpy as np
import cv2
import random

class AugmentData:
    __data_source = ''
    __data_dict = {}
    __augment_dict = {}

    def __init__(self, data_source: str):
        self.__data_source = data_source
        for filename in os.listdir(self.__data_source):
            if filename.endswith(".jpg"):
                age = int(filename.split('_')[0])
                img_path = os.path.join(self.__data_source, filename)
                list_of_imgs = self.__data_dict.get(age, [])
                list_of_imgs.append(img_path)
                self.__data_dict.update({age: list_of_imgs})

    def get_data(self, augment_percent=0):
        train_data, val_data, test_data = self.__prepare_data(augment_percent)
        X_train, Y_train = self.__convert_data(train_data, augment_percent)
        X_val, Y_val = self.__convert_data(val_data)
        X_test, Y_test = self.__convert_data(test_data)
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
        
    
    def __convert_data(self, data_list, augment_percent=0):
        images = []
        ages = []
        for file_path in data_list:
            age = int(file_path[file_path.index('\\')+1::].split('_')[0])  # Pobierz wiek z nazwy pliku
            img = image_processor.preprocess_image_or_frame(file_path, train_data=False)
            images.append(img)
            ages.append(age)
            if augment_percent != 0 and self.__augment_dict.get(age) > 0:
                self.__augment_dict.update({age: self.__augment_dict.get(age)-1})
                images.insert(0, cv2.flip(img, random.randrange(-1, 2)))
                ages.insert(0, age)
        return np.array(images), np.array(ages)

    def __prepare_data(self, augment_percent=0):
        train_data = []
        val_data = []
        test_data = []
        for age in self.__data_dict:
            number_of_pictures = len(self.__data_dict.get(age))
            if number_of_pictures < 5:
                continue
            elif number_of_pictures < 10:
                self.__augment_dict.update({age: 100})
            else:
                self.__augment_dict.update({age: int(augment_percent/100*number_of_pictures)})
            train_amount = int(number_of_pictures*0.7)
            val_amount = int(number_of_pictures*0.2)
            test_amount = int(number_of_pictures*0.1)
            sum = train_amount+val_amount+test_amount
            if sum < number_of_pictures:
                test_amount += number_of_pictures-sum
            self.__get_single_data(self.__data_dict, age, train_amount, val_amount, test_amount, train_data, val_data, test_data)
        return (train_data, val_data, test_data)
    

    def __get_single_data(self, dict: dict, key, train_amount, val_amount, test_amount, train_data: list, val_data: list, test_data: list):
        imgs_list = dict.get(key)

        for x in range(train_amount):
            train_data.append(imgs_list[x])
        for x in range(train_amount, (train_amount+val_amount)):
            val_data.append(imgs_list[x])
        for x in range((train_amount+val_amount), (train_amount+val_amount+test_amount)):
            test_data.append(imgs_list[x])

# todo needed reformat
    def __str__(self):
        result = ''
        for age in self.__data_dict:
            number_of_pictures = len(self.__data_dict.get(age))
            if number_of_pictures < 10:
                # augmenting needed
                continue
            train_amount = int(number_of_pictures*0.7)
            val_amount = int(number_of_pictures*0.2)
            test_amount = int(number_of_pictures*0.1)
            sum = train_amount+val_amount+test_amount
            if sum < number_of_pictures:
                test_amount += number_of_pictures-sum
            result += f'Age: {age}, total amount: {number_of_pictures}\n'
            result += f'Train amount: {train_amount}\n'
            result += f'Validation amount: {val_amount}\n'
            result += f'Test amount: {test_amount}\n\n'
        return result.strip()