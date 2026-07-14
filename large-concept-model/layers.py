import tensorflow as tf
from utils import scaled_dot_product_attention

# ---  Positional Encoding Layer ---
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        positions_range = tf.range(tf.cast(position, tf.float32))
        angle_rads = self.get_angles(
            position=positions_range[:, tf.newaxis],
            i=tf.range(tf.cast(d_model, tf.float32))[tf.newaxis, :],
            d_model=d_model)

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

# ---  Combine Token and Positional Embeddings ---
class TextEmbedder(tf.keras.layers.Layer):
  def __init__(self,text_embedding_layer,positional_encoding_layer,max_seq_len,**kwargs):
    super(TextEmbedder,self).__init__(**kwargs)
    self.text_embedding_layer=text_embedding_layer
    self.positional_encoding_layer=positional_encoding_layer
    self.max_seq_len=max_seq_len
  @tf.function
  def call(self,token_ids):
    token_embeddings=self.text_embedding_layer(token_ids)
    current_seq_len=tf.shape(token_embeddings)[1]
    if current_seq_len<self.max_seq_len:
      padding=tf.zeros([
          tf.shape(token_embeddings)[0],
          self.max_seq_len-current_seq_len,
          tf.shape(token_embeddings)[2]
      ],dtype=token_embeddings.dtype)
    elif current_seq_len>self.max_seq_len:
      token_embeddings=token_embeddings[:,:self.max_seq_len,:]
    combined_embeddings=self.positional_encoding_layer(token_embeddings)
    return combined_embeddings

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

class PointWiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, **kwargs):
        super(PointWiseFeedForwardNetwork, self).__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

class AddNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)

    def call(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)

        self.add_norm1 = AddNormalization()
        self.add_norm2 = AddNormalization()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.add_norm1(x, attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.add_norm2(out1, ffn_output)

        return out2

# ---  Full Transformer Encoder ---
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate, text_embedding_layer, positional_encoding_layer, max_seq_len, text_tokenizer_padded, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.text_tokenizer_padded = text_tokenizer_padded

        self.text_embedder = TextEmbedder(
            text_embedding_layer=text_embedding_layer,
            positional_encoding_layer=positional_encoding_layer,
            max_seq_len=max_seq_len
        )

        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

    @tf.function
    def call(self, raw_text_input, training):
        token_ids = self.text_tokenizer_padded.tokenize(raw_text_input)
        x = self.text_embedder(token_ids)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=None)

        return x


# ---  Graph Neural Network Layers ---
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.feat_transform_dense = tf.keras.layers.Dense(d_model, use_bias=False, name="feature_transform")

        self.attn_dense_per_head = [
            tf.keras.layers.Dense(1, use_bias=False, name=f"attention_head_{h}")
            for h in range(num_heads)
        ]

        self.output_dense = tf.keras.layers.Dense(d_model, name="output_projection")

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    @tf.function
    def call(self, inputs, training=False):
        node_features, adjacency_matrix = inputs

        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]

        transformed_features = self.feat_transform_dense(node_features)

        transformed_features_heads = tf.reshape(
            transformed_features, (batch_size, num_nodes, self.num_heads, self.depth))

        all_head_outputs = []
        all_head_attention_weights = []

        for h in range(self.num_heads):
            current_head_features = transformed_features_heads[:, :, h, :]

            features_i_h = tf.expand_dims(current_head_features, axis=2)
            features_j_h = tf.expand_dims(current_head_features, axis=1)

            concat_ij_h = tf.concat([
                tf.tile(features_i_h, [1, 1, num_nodes, 1]),
                tf.tile(features_j_h, [1, num_nodes, 1, 1])
            ], axis=-1)

            e_h = self.attn_dense_per_head[h](concat_ij_h)
            e_h = tf.nn.leaky_relu(e_h, alpha=0.2)
            e_h = tf.squeeze(e_h, axis=-1)

            attention_mask_h = (1 - adjacency_matrix) * -1e9
            e_masked_h = e_h + attention_mask_h

            attention_weights_h = tf.nn.softmax(e_masked_h, axis=-1)
            attention_weights_h = self.dropout(attention_weights_h, training=training)

            output_h = tf.matmul(attention_weights_h, current_head_features)

            all_head_outputs.append(output_h)
            all_head_attention_weights.append(attention_weights_h)

        output_concat = tf.concat(all_head_outputs, axis=-1)
        output = self.output_dense(output_concat)

        output = self.dropout(output, training=training)
        output = self.layer_norm(node_features + output)

        avg_attention_weights = tf.reduce_mean(tf.stack(all_head_attention_weights, axis=1), axis=1)

        return output, avg_attention_weights

class RelationalGraphConvolutionalLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_relations, dropout_rate=0.1, use_bases=False, num_bases=0, **kwargs):
        super(RelationalGraphConvolutionalLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_relations = num_relations
        self.dropout_rate = dropout_rate
        self.use_bases = use_bases
        self.num_bases = num_bases

        self.self_loop_dense = tf.keras.layers.Dense(d_model, use_bias=False, name="self_loop_transform")

        if self.use_bases:
            if self.num_bases <= 0:
                raise ValueError("num_bases must be > 0 if use_bases is True")
            self.basis_dense_layers = [
                tf.keras.layers.Dense(d_model, use_bias=False, name=f"basis_transform_{i}")
                for i in range(self.num_bases)
            ]
            self.basis_coefficients = self.add_weight(
                name='basis_coeffs',
                shape=(self.num_relations, self.num_bases),
                initializer='glorot_uniform',
                trainable=True
            )
        else:
            self.rel_dense_layers = [
                tf.keras.layers.Dense(d_model, use_bias=False, name=f"relation_transform_{i}")
                for i in range(num_relations)
            ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    @tf.function
    def call(self, inputs, training=False):
        node_features, edge_types, adjacency_matrix = inputs

        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]

        self_loop_messages = self.self_loop_dense(node_features)

        expanded_edge_types = edge_types[:, tf.newaxis, :, :]
        expanded_relation_ids = tf.range(self.num_relations, dtype=tf.int32)[tf.newaxis, :, tf.newaxis, tf.newaxis]
        rel_type_mask = tf.cast(tf.equal(expanded_edge_types, expanded_relation_ids), tf.float32)
        relation_masks = rel_type_mask * adjacency_matrix[:, tf.newaxis, :, :]

        if self.use_bases:
            transformed_basis_features = tf.stack([layer(node_features) for layer in self.basis_dense_layers], axis=0)
            transformed_features_all_rels = tf.einsum('rk,kbnd->rbnd', self.basis_coefficients, transformed_basis_features)
            transformed_features_all_rels = tf.transpose(transformed_features_all_rels, perm=[1, 0, 2, 3])

        else:
            transformed_features_all_rels = tf.stack([layer(node_features) for layer in self.rel_dense_layers], axis=0)
            transformed_features_all_rels = tf.transpose(transformed_features_all_rels, perm=[1, 0, 2, 3])

        messages_from_neighbors_per_rel = tf.einsum('brvu,brud->brvd', relation_masks, transformed_features_all_rels)
        aggregated_messages_from_neighbors = tf.reduce_sum(messages_from_neighbors_per_rel, axis=1)

        output = self_loop_messages + aggregated_messages_from_neighbors
        output = tf.nn.relu(output)
        output = self.dropout(output, training=training)
        output = self.layer_norm(node_features + output)

        return output

class GraphEncoder(tf.keras.layers.Layer):
  def __init__(self,num_layers,d_model,num_relations,dropout_rate=0.1,use_bases=False,num_bases=0.,**kwargs):
    super(GraphEncoder,self).__init__(**kwargs)
    self.num_layers=num_layers
    self.d_model=d_model
    self.num_relations=num_relations
    self.dropout_rate=dropout_rate
    self.use_bases=use_bases
    self.num_bases=num_bases
    self.rgcn_layers=[]
    for i in range(num_layers):
      self.rgcn_layers.append(
          RelationalGraphConvolutionalLayer(
              d_model=d_model,
              num_relations=num_relations,
              dropout_rate=dropout_rate,
              use_bases=use_bases,
              num_bases=num_bases,
              name=f"rgcn_layer_{i}"
          )
      )
  def call(self,inputs,training=False):
    node_features,edge_types,adjacency_matrix=inputs
    x=node_features
    for i in range(self.num_layers):
      x=self.rgcn_layers[i](inputs=(x,edge_types,adjacency_matrix),training=training)
    return x


# ---  Custom Neural Blocks for Concept Composition/Decomposition ---
class ConceptComposer(tf.keras.layers.Layer):
  def __init__(self,d_model,num_heads,dropout_rate=0.1,**kwargs):
    super(ConceptComposer,self).__init__(**kwargs)
    self.d_model=d_model
    self.num_heads=num_heads
    self.input_transform_dense=tf.keras.layers.Dense(d_model,name="input_concept_transform")
    self.attention=MultiHeadAttention(d_model,num_heads)
    self.output_dense=tf.keras.layers.Dense(d_model,name="compose_concept_projection")
    self.layer_norm=tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout=tf.keras.layers.Dropout(dropout_rate)
  def call(self,concept_embeddings,training=False):
    stacked_embeddings=tf.stack(concept_embeddings,axis=1)
    transformed_embeddings=self.input_transform_dense(stacked_embeddings)
    attn_output,_=self.attention(transformed_embeddings,transformed_embeddings,transformed_embeddings,mask=None)
    compose_embedding_mean=tf.reduce_mean(attn_output,axis=1)
    composed_embedding=self.output_dense(compose_embedding_mean)
    initial_mean_embedding=tf.reduce_mean(stacked_embeddings,axis=1)
    composed_embedding=self.layer_norm(initial_mean_embedding+composed_embedding)
    return self.dropout(composed_embedding,training=training)

class ConceptDecomposer(tf.keras.layers.Layer):
  def __init__(self,d_model,num_output_concepts,num_heads,dropout_rate=0.1,**kwargs):
    super(ConceptDecomposer,self).__init__(**kwargs)
    self.d_model=d_model
    self.num_output_concepts=num_output_concepts
    self.num_heads=num_heads
    self.query_vectors=self.add_weight(
      name='concept_queries',
      shape=(num_output_concepts,d_model),
      initializer='glorot_uniform',
      trainable=True
    )
    self.attention=MultiHeadAttention(d_model,num_heads)
    self.output_dense=tf.keras.layers.Dense(d_model,name='decomposed_concept_projection')
    self.layer_norm=tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout=tf.keras.layers.Dropout(dropout_rate)
  def call(self,complex_concept_embedding,training=False):
    batch_size=tf.shape(complex_concept_embedding)[0]
    queries=tf.tile(tf.expand_dims(self.query_vectors,axis=0),[batch_size,1,1])
    complex_concept_expanded=tf.expand_dims(complex_concept_embedding,axis=1)
    attn_output,_=self.attention(complex_concept_expanded,complex_concept_expanded,queries,mask=None)
    decomposed_embeddings=self.output_dense(attn_output)
    decomposed_embeddings=self.layer_norm(queries+decomposed_embeddings)
    decomposed_embeddings=self.dropout(decomposed_embeddings,training=training)
    return decomposed_embeddings