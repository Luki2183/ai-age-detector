import matplotlib.pyplot as plt
import pickle
import keras as kr
from data_augment import AugmentData

augment_percent_to_show = 5

name = "20e_m_" + str(augment_percent_to_show) + "p"

history = pickle.load(open(f"resources/data/history_data_{name}", "rb"))
eval_data = pickle.load(open(f"resources/data/eval_data_{name}", "rb"))

def plot_history(history):
    for key in history:
        if not key.startswith('val_'):
            plt.figure()
            plt.plot(history[key], label=key)
            if f'val_{key}' in history:
                plt.plot(history[f'val_{key}'], label=f'val_{key}')
            plt.xlabel('Epoka')
            plt.legend()
            plt.grid(True)
            plt.show()

plot_history(history)

print(f'{augment_percent_to_show} percent: {eval_data}')

X_train, Y_train, X_val, Y_val, X_test, Y_test = AugmentData('resources/data/UTKFace').get_data(0)

model = kr.models.load_model(f'resources/models/model_{name}.keras')

eval_result = model.evaluate(X_test, Y_test, return_dict=True)