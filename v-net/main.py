from model import build_vnet
import tensorflow as tf

input_shape = (64, 64, 64, 1) 
num_classes = 1 # Assuming binary segmentation

vnet_model = build_vnet(input_shape, num_classes)

vnet_model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_function = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']

vnet_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

print("V-Net model compiled successfully with Adam optimizer, BinaryCrossentropy loss, and accuracy metric.")

num_training_samples = 2
epochs = 2

dummy_train_input = tf.random.normal((num_training_samples,) + input_shape)
dummy_train_target = tf.random.uniform((num_training_samples,) + input_shape[:-1] + (num_classes,), minval=0, maxval=2, dtype=tf.int32)
dummy_train_target = tf.cast(dummy_train_target, tf.float32)

print(f"Dummy training input tensor created with shape: {dummy_train_input.shape}")
print(f"Dummy training target tensor created with shape: {dummy_train_target.shape}")

history = vnet_model.fit(
    dummy_train_input,
    dummy_train_target,
    epochs=epochs,
    batch_size=1,
    verbose=1
)

num_validation_samples = 1

dummy_val_input = tf.random.normal((num_validation_samples,) + input_shape)
dummy_val_target = tf.random.uniform((num_validation_samples,) + input_shape[:-1] + (num_classes,), minval=0, maxval=2, dtype=tf.int32)
dummy_val_target = tf.cast(dummy_val_target, tf.float32)

print(f"Dummy validation input tensor created with shape: {dummy_val_input.shape}")
print(f"Dummy validation target tensor created with shape: {dummy_val_target.shape}")

print("Starting V-Net model evaluation with dummy data...")
evaluation_results = vnet_model.evaluate(
    dummy_val_input,
    dummy_val_target,
    batch_size=1,
    verbose=1
)

print("V-Net model evaluation completed.")
print(f"Evaluation results: {evaluation_results}")

num_inference_samples = 1
dummy_inference_input = tf.random.normal((num_inference_samples,) + input_shape)

print(f"Dummy inference input tensor created with shape: {dummy_inference_input.shape}")

print("Starting V-Net model inference...")
predictions = vnet_model.predict(dummy_inference_input)

print(f"V-Net model inference completed. Prediction output shape: {predictions.shape}")