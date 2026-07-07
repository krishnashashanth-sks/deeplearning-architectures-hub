from tensorflow import keras
from tensorflow.keras import layers

# ---  CNN Feature Extractor Function ---
def create_cnn_feature_extractor(input_shape, num_features):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_features, activation='relu') # Output 'num_features' for fuzzy system
    ])
    return model