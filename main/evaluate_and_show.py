import matplotlib.pyplot as plt
import pickle

history = pickle.load(open("resources/data/history_data", "rb"))
eval_data = pickle.load(open("resources/data/eval_data", "rb"))

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

print(eval_data)