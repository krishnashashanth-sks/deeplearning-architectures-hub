
import tensorflow as tf
import numpy as np
import os
import datetime
from tensorflow import keras
from layers import create_vision_encoder,create_language_encoder,create_policy_transformer,create_vlm_model
from losses import get_loss_function
from inference import rt2_inference
from model import create_rt2_model
from utils import generate_dummy_data

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
EMBEDDING_DIM = 256

# Define parameters for language encoder
VOCAB_SIZE = 10000  # Placeholder vocabulary size
SEQUENCE_LENGTH = 50  # Placeholder maximum sequence length for language instructions
EMBEDDING_DIM_LANG = 256 # Embedding dimension for language, can be different from vision if needed
NUM_HEADS = 4  # Number of attention heads
FF_DIM = 512  # Hidden layer size in feed forward network inside transformer
NUM_TRANSFORMER_BLOCKS = 2 # Number of transformer blocks

# Define parameters for policy transformer
ACTION_DIM = 7  # Example: 6 for end-effector pose (x,y,z,roll,pitch,yaw) + 1 for gripper
NUM_POLICY_TRANSFORMER_BLOCKS = 2  # Number of transformer blocks in the policy head

# Calculate the fused embedding dimension from VLM
FUSED_EMBEDDING_DIM = EMBEDDING_DIM + EMBEDDING_DIM_LANG

policy_loss = get_loss_function(action_type='continuous')
print(f"Policy Loss Function: {policy_loss.name}")

# Define a learning rate schedule
initial_learning_rate = 1e-4
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

# Instantiate the optimizer
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule)
print(f"Optimizer: {optimizer.name} with initial learning rate: {initial_learning_rate}")
# Explicitly enable eager execution to resolve `numpy()` error
tf.config.run_functions_eagerly(True)

#  Refine Distributed Training Strategy Setup
strategy = tf.distribute.get_strategy()
strategy_name = type(strategy).__name__ # Get strategy type name robustly
print(f"Detected distribution strategy: {strategy_name}")

# All model creation and compilation must be within the strategy scope
with strategy.scope():
    # Instantiate base encoders within strategy scope
    # These are needed to create vlm_model and policy_transformer within the scope
    vision_encoder = create_vision_encoder(input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),embedding_dim=EMBEDDING_DIM)

    language_encoder =create_language_encoder(
    vocab_size=VOCAB_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    embed_dim=EMBEDDING_DIM_LANG,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS
)

    # Instantiate VLM and Policy Transformer within strategy scope
    vlm_model = create_vlm_model(vision_encoder, language_encoder, IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS,SEQUENCE_LENGTH,EMBEDDING_DIM)
    # Instantiate the policy transformer
    policy_transformer = create_policy_transformer(FUSED_EMBEDDING_DIM,
        action_dim=ACTION_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_POLICY_TRANSFORMER_BLOCKS)

    # Instantiate the full RT-2 model
    rt2_model = create_rt2_model(vlm_model, policy_transformer, FUSED_EMBEDDING_DIM)
    print("RT-2 Model Summary (created within strategy scope):")
    rt2_model.summary()

    # Compile the model with the defined optimizer and loss function
    # optimizer and policy_loss were defined in the 'Define Loss Functions and Optimization' step
    rt2_model.compile(optimizer=optimizer, loss=policy_loss)

    print(f"RT-2 Model compiled within strategy scope with Optimizer: {optimizer.name} and Loss: {policy_loss.name}")

    # Generate dummy training and validation data
    NUM_TRAINING_SAMPLES = 1000
    NUM_VALIDATION_SAMPLES = 200

    (train_images, train_languages), train_actions = generate_dummy_data(num_samples=NUM_TRAINING_SAMPLES)
    (val_images, val_languages), val_actions = generate_dummy_data(num_samples=NUM_VALIDATION_SAMPLES)

    print(f"Generated {NUM_TRAINING_SAMPLES} dummy training samples and {NUM_VALIDATION_SAMPLES} dummy validation samples.")

    # Create tf.data.Dataset for efficient loading
    train_dataset = tf.data.Dataset.from_tensor_slices(({
        'image_input': train_images,
        'language_input': train_languages
    }, train_actions)).shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(({
        'image_input': val_images,
        'language_input': val_languages
    }, val_actions)).batch(32).prefetch(tf.data.AUTOTUNE)

    print("Created tf.data.Dataset for training and validation.")

    #  Set up Logging and Checkpointing Callbacks (from previous steps)
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_filepath = 'rt2_model_checkpoint/model.weights.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Model checkpoints will be saved to: {checkpoint_filepath}")

    #  Execute Model Training
    print("\nStarting model training...")
    EPOCHS = 5  # Reduced epochs for demonstration

    history = rt2_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, model_checkpoint_callback]
    )

    print("\nModel training finished.")
# Generate a single dummy image and language instruction for inference
dummy_inference_image = np.random.rand(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS).astype(np.float32)
# For the language instruction, in a real scenario, this would be a meaningful string
dummy_inference_language = "move the red block to the left"

print("Running inference with dummy inputs...")
predicted_action = rt2_inference(rt2_model, dummy_inference_image, dummy_inference_language,VOCAB_SIZE,SEQUENCE_LENGTH)

print(f"\nPredicted Robot Actions (7-dimensional vector):\n{predicted_action}")
