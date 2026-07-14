import tensorflow as tf

# ---  Custom Subword Tokenizer ---
class CustomSubwordTokenizer(tf.Module):
    def __init__(self, vocab_size, reserved_tokens=None, name="custom_tokenizer"):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.reserved_tokens = reserved_tokens if reserved_tokens else ["[PAD]", "[UNK]", "[START]", "[END]"]
        self.token_to_id = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(self.reserved_tokens + [f"word_{i}" for i in range(vocab_size - len(self.reserved_tokens))], dtype=tf.string),
                tf.constant(list(range(vocab_size)), dtype=tf.int64)
            ),
            default_value=1 # [UNK] token id
        )
        self.id_to_token = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(list(range(vocab_size)), dtype=tf.int64),
                tf.constant(self.reserved_tokens + [f"word_{i}" for i in range(vocab_size - len(self.reserved_tokens))], dtype=tf.string)
            ),
            default_value="[UNK]"
        )
        self.start_token_id = self.token_to_id.lookup(tf.constant('[START]'))
        self.end_token_id = self.token_to_id.lookup(tf.constant('[END]'))
        self.pad_token_id = self.token_to_id.lookup(tf.constant('[PAD]'))

    @tf.function
    def tokenize(self, texts):
        tokens = tf.strings.split(texts, sep=" ")
        token_ids_ragged = self.token_to_id.lookup(tokens)
        return token_ids_ragged

    @tf.function
    def detokenize(self, token_ids):
        cleaned_token_ids = tf.where(
            tf.logical_and(tf.not_equal(token_ids, self.start_token_id), tf.not_equal(token_ids, self.end_token_id)),
            token_ids, self.pad_token_id)
        tokens = self.id_to_token.lookup(cleaned_token_ids)
        return tf.strings.reduce_join(tokens, separator=" ")

class CustomSubwordTokenizerWithPadding(CustomSubwordTokenizer):
    def __init__(self, vocab_size, max_seq_len, reserved_tokens=None, name="custom_tokenizer"):
        super().__init__(vocab_size=vocab_size, reserved_tokens=reserved_tokens, name=name)
        self.max_seq_len = max_seq_len

    @tf.function
    def tokenize(self, texts):
        token_ids_ragged = super().tokenize(texts)

        start_tokens = tf.cast(tf.fill(tf.shape(token_ids_ragged)[:-1], self.start_token_id), dtype=tf.int64)
        end_tokens = tf.cast(tf.fill(tf.shape(token_ids_ragged)[:-1], self.end_token_id), dtype=tf.int64)
        token_ids_ragged = tf.concat([
            tf.RaggedTensor.from_tensor(tf.expand_dims(start_tokens, axis=-1)),
            token_ids_ragged,
            tf.RaggedTensor.from_tensor(tf.expand_dims(end_tokens, axis=-1))
        ], axis=-1)

        token_ids_padded = token_ids_ragged.to_tensor(default_value=self.pad_token_id, shape=[None, self.max_seq_len])
        token_ids_padded = token_ids_padded[:, :self.max_seq_len]

        return token_ids_padded

