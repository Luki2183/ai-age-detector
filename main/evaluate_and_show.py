import matplotlib.pyplot as plt
import pickle

augment_percent_to_show = 0

history = pickle.load(open(f"resources/data/history_data{augment_percent_to_show}", "rb"))
eval_data = pickle.load(open(f"resources/data/eval_data{augment_percent_to_show}", "rb"))

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