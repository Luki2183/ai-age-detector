from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RandomRotation, RandomZoom, RandomFlip, Input
from data_augment import AugmentData
import pickle

def create_age_estimation_model(input_shape=(200, 200, 3)):
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ])
    model = Sequential([
        # Dla poprawnego wyświetlania parametrów
        Input(shape=input_shape),

        # Augmentacja danych co każdą epoke
        data_augmentation,

        # Pierwsza warstwa konwolucyjna
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Druga warstwa konwolucyjna
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Trzecia warstwa konwolucyjna
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Czwarta warstwa konwolucyjna
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Spłaszczenie
        Flatten(),

        # Warstwy w pełni połączone
        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.3),

        # Warstwa wyjściowa - REGRESJA (pojedyncza wartość wieku)
        Dense(1, activation='linear')
    ])

    return model

model = create_age_estimation_model()

model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)
model.summary()

# Przykładowe dane treningowe
# X_train - zdjęcia (numpy array o kształcie (n_samples, 200, 200, 3))
# Y_train - wiek (numpy array o kształcie (n_samples,))
# augment_percent - procent danych które zostaną augmentowane OGÓLNIE

# OGÓLNY PROCENT AUGMENTACJI DANYCH
augment_percent = 5

# Nazwa zapisywanego pliku (model, history_data, eval_data, best_checkpoint)
name = "20e_m_" + str(augment_percent) + "p"

# 0 procent
X_train, Y_train, X_val, Y_val, X_test, Y_test = AugmentData('resources/data/UTKFace').get_data(augment_percent)

history = model.fit(
    X_train,
    Y_train,
    epochs=20,
    validation_data=(X_val, Y_val),
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=f'resources/data/checkpoints/checkpoint_{name}.keras',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )
    ]
)

model.save(f'resources/models/model_{name}.keras')

eval_result = model.evaluate(X_test, Y_test, return_dict=True)

with open(f'resources/data/history_data_{name}', "wb") as file:
    pickle.dump(history.history, file)

with open(f'resources/data/eval_data_{name}', "wb") as file:
    pickle.dump(eval_result, file)