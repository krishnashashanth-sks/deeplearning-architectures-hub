# ---  Implement inference function ---

def inference(lcm_model_augmented, raw_text_input, initial_graph_node_features, initial_edge_types, initial_adjacency_matrix):
    # Ensure model is in inference mode
    contextualized_text_embeddings, refined_graph_node_features, _, _ = lcm_model_augmented(
        raw_text_input=raw_text_input,
        initial_graph_node_features=initial_graph_node_features,
        initial_edge_types=initial_edge_types,
        initial_adjacency_matrix=initial_adjacency_matrix,
        training=False  # Important: set to False for inference
    )
    return contextualized_text_embeddings, refined_graph_node_features
