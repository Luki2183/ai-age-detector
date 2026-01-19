import os
import shutil

class sortData:
    __data_source = ''
    __data_dict = {}

    def __init__(self, data_source: str):
        self.__data_source = data_source
        for filename in os.listdir(self.__data_source):
            if filename.endswith(".jpg"):
                age = int(filename.split('_')[0])
                img_path = os.path.join(self.__data_source, filename)
                list_of_imgs = self.__data_dict.get(age, [])
                list_of_imgs.append(img_path)
                self.__data_dict.update({age: list_of_imgs})

    def get_data(self):
        train_data = []
        val_data = []
        test_data = []
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


# def augment(dict, key, number_of_pictures):
#     return