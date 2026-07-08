from layers import FractalBlock
from tensorflow import keras
from tensorflow.keras import layers

def build_fractalnet_model(input_shape, num_classes, filters_list, num_columns_list, drop_path_prob_base):
  inputs = keras.Input(shape=input_shape)

  # Initial Convolutional Block
  x = layers.Conv2D(filters_list[0], kernel_size=3, padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)

  # FractalNet Blocks
  for i, (filters, num_columns) in enumerate(zip(filters_list, num_columns_list)):
    # Calculate drop_prob dynamically (e.g., increasing with depth)
    current_drop_prob = drop_path_prob_base * (i + 1) / len(filters_list)
    x = FractalBlock(filters, num_columns, current_drop_prob)(x)
    if i < len(filters_list) - 1: # Add MaxPooling after each block except the last
      x = layers.MaxPooling2D(pool_size=2)(x)

  # Classification Head
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(128, activation='relu')(x) # Example intermediate Dense layer
  outputs = layers.Dense(num_classes, activation='softmax')(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

print("build_fractalnet_model function defined.")