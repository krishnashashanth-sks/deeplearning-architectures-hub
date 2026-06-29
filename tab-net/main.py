from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import TabNet
import tensorflow as tf
import numpy as np
from tensorflow import keras

# 1. Load and Preprocess Dataset (Breast Cancer Dataset)
print("\nLoading and preprocessing Breast Cancer dataset...")
data = load_breast_cancer()
X = data.data
y = data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to TensorFlow tensors
X_train_tf = tf.constant(X_train_scaled, dtype=tf.float32)
y_train_tf = tf.constant(y_train, dtype=tf.int32)
X_test_tf = tf.constant(X_test_scaled, dtype=tf.float32)
y_test_tf = tf.constant(y_test, dtype=tf.int32)

feature_dim = X_train_tf.shape[1]
output_dim = len(np.unique(y_train)) # Number of unique classes

print(f"Feature dimension: {feature_dim}")
print(f"Output dimension (number of classes): {output_dim}")

# 2. Instantiate and Compile TabNet Model
print("\nInstantiating and compiling TabNet model...")

tabnet_model = TabNet(
    feature_dim=feature_dim,
    output_dim=output_dim,
    num_decision_steps=5,
    relaxation_factor=1.5,
    bn_momentum=0.9,
    virtual_batch_size=256, # Ensure this divides your batch_size if using GBN in training
    n_d=16, # Dimension of the decision layer output
    n_a=16  # Dimension of the attention layer output
)

# Build the model by passing a dummy input
tabnet_model.build(input_shape=(None, feature_dim))

tabnet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(), # Use SparseCategoricalCrossentropy for integer labels
    metrics=['accuracy']
)

print("TabNet Model Summary:")
tabnet_model.summary()

# 3. Train the Model
print("\nTraining TabNet model...")
history = tabnet_model.fit(
    X_train_tf,
    y_train_tf,
    epochs=10, # Reduced epochs for demonstration
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# 4. Evaluate and Infer
print("\nEvaluating TabNet model on test data...")
loss, accuracy = tabnet_model.evaluate(X_test_tf, y_test_tf, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

print("\nPerforming inference on test data (first 5 samples)...")
predictions = tabnet_model.predict(X_test_tf[:5])
print("Raw predictions (probabilities):\n", predictions)
predicted_classes = tf.argmax(predictions, axis=1).numpy()
print("Predicted classes:\n", predicted_classes)
print("True classes:\n", y_test_tf[:5].numpy())