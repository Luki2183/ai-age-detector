from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from data_augment import AugmentData
import pickle

def create_age_estimation_model(input_shape=(200, 200, 3)):
    model = Sequential([
        # Pierwsza warstwa konwolucyjna
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
    loss='mean_absolute_error',
    metrics=['mae']
)
model.summary()

# Przykładowe dane treningowe
# X_train - zdjęcia (numpy array o kształcie (n_samples, 200, 200, 3))
# Y_train - wiek (numpy array o kształcie (n_samples,))

X_train, Y_train, X_val, Y_val, X_test, Y_test = AugmentData('resources/UTKFaceSmall').get_data()

history = model.fit(
    X_train,
    Y_train,
    epochs=3,
    validation_data=(X_val, Y_val),
    verbose=1
    # callbacks=[
    #     keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    # ]
)

model.save('resources/models/test.keras')

eval_result = model.evaluate(X_test, Y_test, return_dict=True)

with open('resources/data/history_data', "wb") as file:
    pickle.dump(history.history, file)

with open('resources/data/eval_data', "wb") as file:
    pickle.dump(eval_result, file)