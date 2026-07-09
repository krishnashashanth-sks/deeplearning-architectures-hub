def tokenize_corpus(corpus):
    tokenized_output = []
    for sentence in corpus:
        # Convert to lowercase and split by whitespace
        tokenized_sentence = sentence.lower().split()
        tokenized_output.append(tokenized_sentence)
    return tokenized_output

def numericalize_and_pad_sequences(tokenized_sentences, word_to_idx, max_seq_len):
    numericalized_sequences = []
    pad_id = word_to_idx["<pad>"]
    unk_id = word_to_idx["<unk>"]

    for sentence_tokens in tokenized_sentences:
        numerical_sequence = [
            word_to_idx.get(word, unk_id) for word in sentence_tokens
        ]

        # Pad or truncate
        if len(numerical_sequence) < max_seq_len:
            numerical_sequence += [pad_id] * (max_seq_len - len(numerical_sequence))
        elif len(numerical_sequence) > max_seq_len:
            numerical_sequence = numerical_sequence[:max_seq_len]

        numericalized_sequences.append(numerical_sequence)
    return numericalized_sequences