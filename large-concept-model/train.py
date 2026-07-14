import tensorflow as tf
from losses import *
import time

# ---  Training Step Function ---
@tf.function
def train_step(
    lcm_model_augmented,
    optimizer,
    l2_reg_rate,
    lr_var,
    learning_rate_schedule,
    raw_text_input,
    initial_graph_node_features,
    initial_edge_types,
    initial_adjacency_matrix,

    true_text_token_ids_for_mlm,
    text_mask_indices_for_mlm,

    true_graph_node_features_for_rec,
    graph_node_mask_indices_for_rec,

    query_embeddings_infonce,
    positive_key_embeddings_infonce,
    negative_key_embeddings_infonce,

    kg_head_embeddings,
    kg_relation_embeddings,
    kg_positive_tail_embeddings,
    kg_negative_tail_embeddings
):
    with tf.GradientTape() as tape:
        contextualized_text_embeddings, \
        refined_graph_node_features, \
        mlm_predicted_logits, \
        graph_rec_predicted_embeddings = lcm_model_augmented(
            raw_text_input=raw_text_input,
            initial_graph_node_features=initial_graph_node_features,
            initial_edge_types=initial_edge_types,
            initial_adjacency_matrix=initial_adjacency_matrix,
            training=True
        )

        mlm_loss = masked_language_model_loss(
            predicted_logits=mlm_predicted_logits,
            true_token_ids=true_text_token_ids_for_mlm,
            mask_indices=text_mask_indices_for_mlm
        )

        graph_rec_loss = masked_graph_reconstruction_loss(
            predicted_embeddings=graph_rec_predicted_embeddings,
            true_embeddings=true_graph_node_features_for_rec,
            mask_indices=graph_node_mask_indices_for_rec
        )

        infonce_loss_value = infonce_loss(
            query_embeddings=query_embeddings_infonce,
            positive_key_embeddings=positive_key_embeddings_infonce,
            negative_key_embeddings=negative_key_embeddings_infonce
        )

        kg_triplet_loss_value = transe_triplet_loss(
            head_embeddings=kg_head_embeddings,
            relation_embeddings=kg_relation_embeddings,
            positive_tail_embeddings=kg_positive_tail_embeddings,
            negative_tail_embeddings=kg_negative_tail_embeddings
        )

        loss_weights = {
            'mlm': 1.0,
            'graph_rec': 1.0,
            'infonce': 0.5,
            'kg_triplet': 0.8
        }

        total_loss = (
            loss_weights['mlm'] * mlm_loss +
            loss_weights['graph_rec'] * graph_rec_loss +
            loss_weights['infonce'] * infonce_loss_value +
            loss_weights['kg_triplet'] * kg_triplet_loss_value
        )

        l2_loss = 0.0
        for var in lcm_model_augmented.trainable_variables:
            if 'kernel' in var.name or 'embedding' in var.name:
                l2_loss += tf.nn.l2_loss(var)
        total_loss += l2_reg_rate * l2_loss

    gradients = tape.gradient(total_loss, lcm_model_augmented.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

    current_learning_rate_from_schedule = learning_rate_schedule(optimizer.iterations)
    lr_var.assign(current_learning_rate_from_schedule)

    optimizer.apply_gradients(zip(gradients, lcm_model_augmented.trainable_variables))

    return total_loss, {
        'mlm_loss': mlm_loss,
        'graph_rec_loss': graph_rec_loss,
        'infonce_loss': infonce_loss_value,
        'kg_triplet_loss': kg_triplet_loss_value,
        'l2_reg_loss': l2_reg_rate * l2_loss
    }

    # --- 16. Implement train_model function ---

def train_model(
    lcm_model_augmented,
    optimizer,
    learning_rate_schedule,
    lr_var,
    epochs,
    steps_per_epoch,
    batch_size,
    num_nodes_graph,
    num_unique_relations,
    d_model,
    text_tokenizer_padded,
    l2_reg_rate,
    mlm_mask_prob=0.15,
    graph_mask_prob=0.2,
    num_negative_samples=5
):
    print(f"\nStarting training for {epochs} epochs, {steps_per_epoch} steps/epoch.")

    # Dummy data for demonstration purposes, would be replaced by actual data pipeline
    # in a real scenario.
    def generate_dummy_batch():
        # Raw text input
        raw_text_input = tf.constant([
            "The quick brown fox jumps over the lazy dog. This is a sample sentence.",
            "Artificial intelligence is transforming many industries globally.",
            "Quantum computing promises to solve previously intractable problems.",
            "Concepts are formed through experience and interaction with the world."
        ])
        # Randomly select a batch of texts
        indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=tf.shape(raw_text_input)[0], dtype=tf.int32)
        batch_raw_text_input = tf.gather(raw_text_input, indices)

        # Initial graph inputs
        batch_initial_graph_node_features = tf.random.uniform(shape=(batch_size, num_nodes_graph, d_model), dtype=tf.float32)
        batch_initial_edge_types = tf.random.uniform(shape=(batch_size, num_nodes_graph, num_nodes_graph), minval=0, maxval=num_unique_relations, dtype=tf.int32)
        batch_initial_adjacency_matrix = tf.cast(tf.random.uniform(shape=(batch_size, num_nodes_graph, num_nodes_graph), minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
        # Make adjacency matrix symmetric and remove self-loops for graph demo
        batch_initial_adjacency_matrix = tf.cast((batch_initial_adjacency_matrix + tf.transpose(batch_initial_adjacency_matrix, perm=[0, 2, 1])) > 0, dtype=tf.float32)
        identity_matrix_graph = tf.eye(num_nodes_graph, batch_shape=[batch_size], dtype=tf.float32)
        batch_initial_adjacency_matrix = batch_initial_adjacency_matrix - identity_matrix_graph
        batch_initial_adjacency_matrix = tf.nn.relu(batch_initial_adjacency_matrix)

        # Targets/Masks for MLM Loss
        batch_true_text_token_ids_for_mlm = text_tokenizer_padded.tokenize(batch_raw_text_input)
        random_mask_indices_text = tf.random.uniform(shape=tf.shape(batch_true_text_token_ids_for_mlm), minval=0., maxval=1.)
        batch_text_mask_indices_for_mlm = random_mask_indices_text < mlm_mask_prob

        # Targets/Masks for Graph Reconstruction Loss (node features)
        batch_true_graph_node_features_for_rec = tf.random.uniform(shape=(batch_size, num_nodes_graph, d_model), dtype=tf.float32)
        random_mask_indices_graph = tf.random.uniform(shape=(batch_size, num_nodes_graph), minval=0., maxval=1.)
        batch_graph_node_mask_indices_for_rec = random_mask_indices_graph < graph_mask_prob

        # Inputs for InfoNCE Loss
        # In a real scenario, these would be derived from model outputs, e.g., (text_embedding, graph_node_embedding, other_graph_node_embedding)
        batch_query_embeddings_infonce = tf.random.uniform(shape=(batch_size, d_model), dtype=tf.float32)
        batch_positive_key_embeddings_infonce = tf.random.uniform(shape=(batch_size, d_model), dtype=tf.float32)
        batch_negative_key_embeddings_infonce = tf.random.uniform(shape=(batch_size, num_negative_samples, d_model), dtype=tf.float32)

        # Inputs for KG Triplet Loss
        # In a real scenario, these would be sampled from knowledge graph triplets
        batch_kg_head_embeddings = tf.random.uniform(shape=(batch_size, d_model), dtype=tf.float32)
        batch_kg_relation_embeddings = tf.random.uniform(shape=(batch_size, d_model), dtype=tf.float32)
        batch_kg_positive_tail_embeddings = tf.random.uniform(shape=(batch_size, d_model), dtype=tf.float32)
        batch_kg_negative_tail_embeddings = tf.random.uniform(shape=(batch_size, d_model), dtype=tf.float32)

        return (
            batch_raw_text_input, batch_initial_graph_node_features, batch_initial_edge_types, batch_initial_adjacency_matrix,
            batch_true_text_token_ids_for_mlm, batch_text_mask_indices_for_mlm,
            batch_true_graph_node_features_for_rec, batch_graph_node_mask_indices_for_rec,
            batch_query_embeddings_infonce, batch_positive_key_embeddings_infonce, batch_negative_key_embeddings_infonce,
            batch_kg_head_embeddings, batch_kg_relation_embeddings, batch_kg_positive_tail_embeddings, batch_kg_negative_tail_embeddings
        )

    for epoch in range(epochs):
        start_time = time.time()
        total_epoch_loss = 0.0
        total_mlm_loss = 0.0
        total_graph_rec_loss = 0.0
        total_infonce_loss = 0.0
        total_kg_triplet_loss = 0.0
        total_l2_loss = 0.0

        for step in range(steps_per_epoch):
            # Update learning rate (handled by lr_var assignment in train_step)
            # current_lr = learning_rate_schedule(optimizer.iterations).numpy()

            batch_data = generate_dummy_batch()
            total_loss, individual_losses = train_step(
                *batch_data # Unpack the tuple of dummy batch data
            )

            total_epoch_loss += total_loss.numpy()
            total_mlm_loss += individual_losses['mlm_loss'].numpy()
            total_graph_rec_loss += individual_losses['graph_rec_loss'].numpy()
            total_infonce_loss += individual_losses['infonce_loss'].numpy()
            total_kg_triplet_loss += individual_losses['kg_triplet_loss'].numpy()
            total_l2_loss += individual_losses['l2_reg_loss'].numpy()

            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}: "
                      f"Total Loss: {total_epoch_loss / (step+1):.4f}, "
                      f"MLM Loss: {total_mlm_loss / (step+1):.4f}, "
                      f"Graph Rec Loss: {total_graph_rec_loss / (step+1):.4f}, "
                      f"InfoNCE Loss: {total_infonce_loss / (step+1):.4f}, "
                      f"KG Triplet Loss: {total_kg_triplet_loss / (step+1):.4f}, "
                      f"L2 Loss: {total_l2_loss / (step+1):.4f}"
                      f" LR: {lr_var.numpy():.7f}")

        end_time = time.time()
        print(f"Epoch {epoch+1} finished in {end_time - start_time:.2f} seconds. Average Total Loss: {total_epoch_loss / steps_per_epoch:.4f}")
