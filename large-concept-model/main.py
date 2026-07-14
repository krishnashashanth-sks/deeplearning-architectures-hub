import tensorflow as tf
from tokenizer import CustomSubwordTokenizerWithPadding
from layers import PositionalEncoding
from model import LCMModel, AugmentedLCMModel
from utils import LinearWarmupCosineDecay
from train import train_model
from inference import inference

# ---  Define Core Hyperparameters ---
d_model = 512  # Dimensionality of the model's embeddings
max_seq_len = 256  # Maximum sequence length for linearized text input
vocab_size_text = 30000  # Size of the vocabulary for subword tokenizer (e.g., BPE, WordPiece)

# Placeholder for number of unique concepts and relations. These would be determined
# from the generated/acquired dataset.
num_unique_concepts = 100000 # Example: 100k distinct concepts
num_unique_relations = 200    # Example: 200 distinct relation types

# ---  Text Embedding Layer ---
text_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size_text, output_dim=d_model)

concept_embedding_layer = tf.keras.layers.Embedding(
    input_dim=num_unique_concepts,
    output_dim=d_model,
    name="concept_embedding"
)
relation_embedding_layer = tf.keras.layers.Embedding(
    input_dim=num_unique_relations,
    output_dim=d_model,
    name="relation_embedding"
)

# --- Initialization of objects for training ---
text_tokenizer_padded = CustomSubwordTokenizerWithPadding(vocab_size=vocab_size_text, max_seq_len=max_seq_len)
positional_encoding_layer = PositionalEncoding(position=max_seq_len, d_model=d_model)

# Hyperparameters for Transformer and Graph encoders
num_heads = 8
dff = d_model * 4
num_transformer_layers = 4
graph_encoder_num_layers = 2
graph_encoder_use_bases = False
graph_encoder_num_bases = 30 # Only relevant if graph_encoder_use_bases is True

lcm_base_model = LCMModel(
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    transformer_num_layers=num_transformer_layers,
    graph_encoder_num_layers=graph_encoder_num_layers,
    num_unique_relations=num_unique_relations,
    text_embedding_layer=text_embedding_layer,
    positional_encoding_layer=positional_encoding_layer,
    max_seq_len=max_seq_len,
    text_tokenizer_padded=text_tokenizer_padded,
    transformer_dropout_rate=0.1,
    graph_dropout_rate=0.1,
    graph_use_bases=graph_encoder_use_bases,
    graph_num_bases=graph_encoder_num_bases
)
lcm_model_augmented = AugmentedLCMModel(lcm_base_model=lcm_base_model, vocab_size_text=vocab_size_text, d_model=d_model)

# Learning rate schedule setup
peak_learning_rate = 1e-4
total_epochs = 100
steps_per_epoch = 1000
total_training_steps = total_epochs * steps_per_epoch
warmup_epochs = 10
warmup_steps = warmup_epochs * steps_per_epoch

learning_rate_schedule = LinearWarmupCosineDecay(
    peak_learning_rate=peak_learning_rate,
    warmup_steps=warmup_steps,
    total_training_steps=total_training_steps
)

lr_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_var # Pass the tf.Variable here
)

# L2 regularization rate
l2_reg_rate = 1e-5

batch_size = 2 # Example batch size
num_nodes_graph = 10 # Example number of nodes in graph

# Define training parameters for the demo
demo_epochs = 2 # Reduced epochs for quicker demonstration
demo_steps_per_epoch = 5 # Reduced steps per epoch

train_model(
    lcm_model_augmented=lcm_model_augmented,
    optimizer=optimizer,
    learning_rate_schedule=learning_rate_schedule,
    lr_var=lr_var,
    epochs=demo_epochs,
    steps_per_epoch=demo_steps_per_epoch,
    batch_size=batch_size,
    num_nodes_graph=num_nodes_graph,
    num_unique_relations=num_unique_relations,
    d_model=d_model,
    text_tokenizer_padded=text_tokenizer_padded,
    l2_reg_rate=l2_reg_rate
)

# Example inference inputs
inference_raw_text = tf.constant(["The cat sat on the mat.", "Birds fly in the sky."])
inference_graph_node_features = tf.random.uniform(shape=(2, 5, d_model), dtype=tf.float32) # Smaller graph for demo
inference_edge_types = tf.random.uniform(shape=(2, 5, 5), minval=0, maxval=num_unique_relations, dtype=tf.int32)
inference_adjacency_matrix = tf.cast(tf.random.uniform(shape=(2, 5, 5), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)

# Get predictions
contextualized_text_output, refined_graph_output = inference(
    lcm_model_augmented,
    inference_raw_text,
    inference_graph_node_features,
    inference_edge_types,
    inference_adjacency_matrix
)

print(f"Inference - Contextualized Text Output shape: {contextualized_text_output.shape}")
print(f"Inference - Refined Graph Output shape: {refined_graph_output.shape}")