from algorithm import km_algorithm
from layers import  
# --- Configuration Parameters ---
input_shape = (28, 28, 1) # Example for image input (e.g., MNIST)
num_extracted_features = 128 # Output features from CNN
num_linguistic_terms = 3 # For fuzzification (e.g., Low, Medium, High)
num_output_classes = 10 # Example for classification

print("Configuration:")
print(f"  Input Shape: {input_shape}")
print(f"  CNN Extracted Features: {num_extracted_features}")
print(f"  Number of Linguistic Terms: {num_linguistic_terms}")
print(f"  Number of Output Classes: {num_output_classes}\n")

# --- 1. Conceptual Type-2 Fuzzification & Visualization ---
print("--- Demonstrating Conceptual Type-2 Fuzzification ---")
x_values = np.linspace(0, 10, 100)
center_val = 5
width_val = 6
uncertainty = 0.2
feature_value = 5.5 # A crisp feature value from the CNN

umf_params, lmf_params = create_type2_trapezoidal_mf_params(center_val, width_val, uncertainty)
umf_memberships = np.array([trapezoidal_mf(x, *umf_params) for x in x_values])
lmf_memberships = np.array([trapezoidal_mf(x, *lmf_params) for x in x_values])

plot_type2_mf(x_values, umf_memberships, lmf_memberships, feature_value=feature_value)
print("Conceptual Type-2 Membership Function plotted.\n")

# --- 2. Karnik-Mendel Algorithm Example & Visualization ---
print("--- Demonstrating Karnik-Mendel (KM) Algorithm ---")
# Apply the KM algorithm to the conceptual Type-2 MF
cL, cR = km_algorithm(x_values, umf_memberships, lmf_memberships)
print(f"Calculated centroid interval [cL, cR]: [{cL:.4f}, {cR:.4f}]")

plot_type2_mf(x_values, umf_memberships, lmf_memberships, title='Type-2 Trapezoidal MF with KM Centroid')
plt.axvline(x=cL, color='green', linestyle='-', label=f'Centroid Left (cL): {cL:.2f}')
plt.axvline(x=cR, color='purple', linestyle='-', label=f'Centroid Right (cR): {cR:.2f}')
plt.legend()
plt.show()
print("KM Centroid plotted.\n")

# --- 3. Defuzzification (for Type-2) ---
print("--- Demonstrating Defuzzification ---")
defuzzified_output = (cL + cR) / 2
print(f"Defuzzified output from KM interval: {defuzzified_output:.4f}\n")

# --- 4. Hybrid Model Integration ---
print("--- Building the Hybrid CNN-Fuzzification Model ---")
input_layer = keras.Input(shape=input_shape)

# CNN Feature Extraction
cnn_extractor = create_cnn_feature_extractor(input_shape, num_extracted_features)
cnn_features = cnn_extractor(input_layer)

# Fuzzification Layer (Type-1 example)
fuzzified_output = FuzzificationLayer(num_linguistic_terms=num_linguistic_terms, input_dim=num_extracted_features)(cnn_features)

# Placeholder for Fuzzy Inference, Aggregation, Type-Reduction, Defuzzification
# For this simple example, we flatten and use a dense layer for classification.
flattened_fuzzy_output = layers.Flatten()(fuzzified_output)
final_output = layers.Dense(num_output_classes, activation='softmax')(flattened_fuzzy_output)

hybrid_model = keras.Model(inputs=input_layer, outputs=final_output)
hybrid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hybrid_model.summary()
print("Hybrid Model built and compiled.\n")

# --- 5. Training the Hybrid Model (Example with Dummy Data) ---
print("--- Training the Hybrid Model with Dummy Data ---")
num_samples = 100
dummy_images = np.random.rand(num_samples, input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
dummy_labels = keras.utils.to_categorical(np.random.randint(0, num_output_classes, num_samples), num_classes=num_output_classes)

print(f"Dummy images shape: {dummy_images.shape}")
print(f"Dummy labels shape: {dummy_labels.shape}")

history = hybrid_model.fit(
    dummy_images,
    dummy_labels,
    epochs=5, # Train for a few epochs
    batch_size=32,
    validation_split=0.2 # Use a small validation split
)
print("\nModel training complete.\n")

# --- 6. Model Inference ---
print("--- Performing Model Inference with Dummy Data ---")
num_inference_samples = 5
dummy_inference_images = np.random.rand(num_inference_samples, input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)

print(f"Dummy inference images shape: {dummy_inference_images.shape}")

predictions = hybrid_model.predict(dummy_inference_images)

print("\nModel predictions:")
predicted_classes = np.argmax(predictions, axis=1)

for i in range(num_inference_samples):
    print(f"Sample {i+1}: Predicted class = {predicted_classes[i]} (Probabilities: {predictions[i]})")
print("\n--- Hybrid CNN-Fuzzy System Demonstration Complete ---")
