import tensorflow as tf
from layers import TransformerEncoder,GraphEncoder

# ---  Integrated Neural Processing Modules ---
class LCMModel(tf.keras.Model):
  def __init__(self,d_model,
               num_heads,
               dff,
               transformer_num_layers,
               graph_encoder_num_layers,
               num_unique_relations,
               text_embedding_layer,
               positional_encoding_layer,
               max_seq_len,
               text_tokenizer_padded,
               transformer_dropout_rate=0.1,
               graph_dropout_rate=0.1,
               graph_use_bases=False,
               graph_num_bases=30,
               **kwargs):
    super(LCMModel,self).__init__(**kwargs)
    self.transformed_encoder=TransformerEncoder(
        num_layers=transformer_num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,rate=transformer_dropout_rate,
        text_embedding_layer=text_embedding_layer,
        positional_encoding_layer=positional_encoding_layer,
        max_seq_len=max_seq_len,
        text_tokenizer_padded=text_tokenizer_padded,
        name='text_transformer_encoder'
    )
    self.graph_encoder=GraphEncoder(
        num_layers=graph_encoder_num_layers,
        d_model=d_model,
        num_relations=num_unique_relations,
        dropout_rate=graph_dropout_rate,
        use_bases=graph_use_bases,
        num_bases=graph_num_bases,
        name="concept_graph_encoder"
    )
  def call(self,raw_text_input,initial_graph_node_features,initial_edge_types,initial_adjacency_matrix,training=False):
    contextualized_text_embeddings=self.transformed_encoder(raw_text_input,training=training)
    refined_graph_node_features=self.graph_encoder(
        inputs=(initial_graph_node_features,initial_edge_types,initial_adjacency_matrix),training=training
    )
    return contextualized_text_embeddings,refined_graph_node_features


# ---  Augmented LCMModel with Prediction Heads ---
class AugmentedLCMModel(tf.keras.Model):
    def __init__(self, lcm_base_model, vocab_size_text, d_model, **kwargs):
        super(AugmentedLCMModel, self).__init__(**kwargs)
        self.lcm_base_model = lcm_base_model

        self.mlm_head = tf.keras.layers.Dense(vocab_size_text, name="mlm_prediction_head")
        self.graph_rec_head = tf.keras.layers.Dense(d_model, name="graph_reconstruction_head")

    def call(self, raw_text_input, initial_graph_node_features, initial_edge_types, initial_adjacency_matrix, training=False):
        contextualized_text_embeddings, refined_graph_node_features = self.lcm_base_model(
            raw_text_input=raw_text_input,
            initial_graph_node_features=initial_graph_node_features,
            initial_edge_types=initial_edge_types,
            initial_adjacency_matrix=initial_adjacency_matrix,
            training=training
        )

        mlm_predicted_logits = self.mlm_head(contextualized_text_embeddings)
        graph_rec_predicted_embeddings = self.graph_rec_head(refined_graph_node_features)

        return (
            contextualized_text_embeddings,
            refined_graph_node_features,
            mlm_predicted_logits,
            graph_rec_predicted_embeddings
        )
