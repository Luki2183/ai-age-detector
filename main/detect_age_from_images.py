import keras as kr
import os
import age_prediction


model = kr.models.load_model('resources/models/age_estimation_model50e.keras')

data_dir = 'resources/data/tomy'

for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            age = age_prediction.predict_age(model, img_path)
            print(f'Przewidywany wiek: {age:.1f} lat zdjęcia {filename}')