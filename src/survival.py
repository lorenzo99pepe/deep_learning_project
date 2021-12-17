import numpy as np
import tensorflow

from tensorflow import keras


def process_and_tensorize(image_list_clean, age, y):
    X = []
    for i in range(len(image_list_clean)):
        img = np.load(image_list_clean[i])
        centered = img - np.mean(img)
        if np.std(centered) != 0:
            standardized = centered / np.std(centered)
        stacked = np.stack((standardized,) * 3, axis=-1)
        X.append(stacked)
    
    X_ten = keras.backend.constant(X)
    age_ten = keras.backend.constant(age)
    target = keras.backend.constant(np.array([int(lab) for lab in y]))

    return X_ten, age_ten, target


def build_model(input_img, age):
    base_model = keras.applications.resnet50.ResNet50(
        weights="imagenet", include_top=False, input_tensor=input_img
    )
    a = keras.layers.GlobalAveragePooling2D()(base_model.output)
    a = keras.layers.concatenate([a, age])
    a = keras.layers.BatchNormalization()(a)
    output = keras.layers.Dense(1)(a)

    model = keras.models.Model(inputs=[input_img, age], outputs=output)
    return model, base_model
