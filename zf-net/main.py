from model import build_zfnet_advanced
from tensorflow.keras import optimizers
from dataset import IMG_HEIGHT,IMG_WIDTH,num_classes_cifar100,train_ds,test_ds,x_test
from inference import predict_image
from tensorflow.keras.preprocessing.image import save_img

model = build_zfnet_advanced(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=num_classes_cifar100)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model recompiled for CIFAR-100 with 100 classes.")
model.summary()

epochs = 10 # You can increase this for more training

print(f"\nStarting training for {epochs} epochs...")
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=test_ds
)


# Let's create a dummy image for demonstration purposes if no image is available
# In a real scenario, you would have a local image file.

dummy_image_path = 'dummy_cifar100_test_image.png'
dummy_image = x_test[0] # Take the first image from the test set
save_img(dummy_image_path, dummy_image) # Save it as a file

print("\n--- Inference Example ---")
predicted_idx, conf, all_predictions = predict_image(model, dummy_image_path, IMG_HEIGHT, IMG_WIDTH)

cifar100_class_names = [f'class_{i}' for i in range(num_classes_cifar100)] # Placeholder, actual names require mapping
predicted_idx, conf, _ = predict_image(model, dummy_image_path, IMG_HEIGHT, IMG_WIDTH, class_names=cifar100_class_names)