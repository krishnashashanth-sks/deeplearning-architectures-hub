import numpy as np
np.random.seed(42)
import tensorflow as tf
 
# Define vocabulary sizes and embedding dimensions
NUM_USERS = 1000
NUM_ITEMS = 5000
NUM_CATEGORIES = 100
EMBEDDING_DIM = 32
SEQ_LEN = 10  # Max historical sequence length

# Generate synthetic data
def generate_data(num_samples):
    user_ids = np.random.randint(0, NUM_USERS, num_samples)
    item_ids = np.random.randint(0, NUM_ITEMS, num_samples)
    category_ids = np.random.randint(0, NUM_CATEGORIES, num_samples)

    # Historical sequences
    hist_item_ids = np.random.randint(0, NUM_ITEMS, (num_samples, SEQ_LEN))
    hist_category_ids = np.random.randint(0, NUM_CATEGORIES, (num_samples, SEQ_LEN))

    # Simulate user behavior: make some historical items related to the candidate item
    for i in range(num_samples):
        if np.random.rand() < 0.5: # 50% chance to have a relevant item
            relevant_idx = np.random.randint(0, SEQ_LEN)
            hist_item_ids[i, relevant_idx] = item_ids[i] # Candidate item in history
            hist_category_ids[i, relevant_idx] = category_ids[i]

    labels = np.random.randint(0, 2, num_samples) # Binary labels (click/no click)

    return {
        'user_id': user_ids,
        'item_id': item_ids,
        'category_id': category_ids,
        'hist_item_ids': hist_item_ids,
        'hist_category_ids': hist_category_ids
    }, labels

NUM_SAMPLES = 10000
x_train, y_train = generate_data(NUM_SAMPLES)
x_val, y_val = generate_data(NUM_SAMPLES // 5)

# Instantiate the model
din_model = DIN(
    num_users=NUM_USERS,
    num_items=NUM_ITEMS,
    num_categories=NUM_CATEGORIES,
    embedding_dim=EMBEDDING_DIM,
    seq_len=SEQ_LEN
)

# Compile the model
din_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Build the model with sample input shapes to see summary
din_model.build({
    'user_id': tf.TensorShape([None]),
    'item_id': tf.TensorShape([None]),
    'category_id': tf.TensorShape([None]),
    'hist_item_ids': tf.TensorShape([None, SEQ_LEN]),
    'hist_category_ids': tf.TensorShape([None, SEQ_LEN])
})

history = din_model.fit(
    x_train, y_train,
    batch_size=256,
    epochs=5,
    validation_data=(x_val, y_val)
)

loss, accuracy, auc = din_model.evaluate(x_val, y_val, verbose=0)

# Generate a small batch of synthetic data for inference
num_inference_samples = 5
x_inference, _ = generate_data(num_inference_samples)

print("Input features for inference:")
for k, v in x_inference.items():
    print(f"{k}: {v.shape}")

# Make predictions
predictions = din_model.predict(x_inference)

print("\nModel predictions (CTR scores):")
print(predictions)