from model import build_siamese_net,SiameseModel
from dataset import train_dataset,IMG_HEIGHT,IMG_WIDTH,test_dataset,test_triplets,x_test
import tensorflow as tf
from evaluate import evaluate_model
from inference import inference_function

siamese_network=build_siamese_net(IMG_HEIGHT,IMG_WIDTH)

learning_rate = 0.001
epochs = 10

# 1. Instantiate the SiameseModel class
siamese_trainer = SiameseModel(siamese_network=siamese_network)

# 2. Compile the siamese_trainer model
siamese_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

history = siamese_trainer.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)
sample_image = x_test[0]
print(f"Sample image shape: {sample_image.shape}")

# Get its embedding
sample_embedding = inference_function(sample_image,siamese_trainer.siamese_network)

model_accuracy = evaluate_model(siamese_trainer, test_triplets)

print(f"Model Accuracy on Test Triplet Set: {model_accuracy:.2f}%")
