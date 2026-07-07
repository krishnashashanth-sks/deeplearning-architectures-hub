import tensorflow as tf

# Text Tokenization and Vocabulary
def build_tokenizer_and_vocab(captions):
    # 1. First, create a temporary tokenizer to determine the natural vocabulary of the captions.
    temp_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=None,
        filters='', # No filters here, as <start> and <end> are already in captions
        lower=False # Keep case as is since <start> and <end> are mixed
    )
    temp_tokenizer.fit_on_texts(captions)

    # 2. Collect all unique words from the captions
    words_from_captions = set(temp_tokenizer.word_index.keys())

    # 3. Define all tokens, including special ones, in desired order
    # This order ensures <pad> is 0, <unk> is 1, <start> is 2, <end> is 3
    # We'll use <unk> before <start>/<end> for consistency with some tutorials.
    all_tokens_ordered = ['<pad>', '<unk>', '<start>', '<end>'] # Explicitly assign known special tokens

    # Add unique words from captions, maintaining alphabetical order for consistency (optional but good practice)
    # Exclude special tokens that are already added
    for word in sorted(list(words_from_captions)):
        if word not in all_tokens_ordered:
            all_tokens_ordered.append(word)

    # 4. Create a new tokenizer instance
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=None, # Use all words
        oov_token="<unk>", # This ensures any word not in `word_index` maps to <unk>
        filters='', # No filters, we pre-processed captions
        lower=False # Keep case
    )

    # Manually build the word_index and index_word for the tokenizer using the defined order
    tokenizer.word_index = {word: i for i, word in enumerate(all_tokens_ordered)}
    tokenizer.index_word = {i: word for i, word in enumerate(all_tokens_ordered)}

    # For robustness, update _num_words and _vocab_size manually if they are accessed internally
    tokenizer._num_words = len(all_tokens_ordered)
    tokenizer.word_docs = temp_tokenizer.word_docs # Copy word_docs from temp_tokenizer for correct counts
    tokenizer.word_counts = temp_tokenizer.word_counts # Copy word_counts

    # 5. Convert captions to sequences using this new tokenizer
    # The tokenizer's internal `word_index` is now correctly set.
    train_seqs = tokenizer.texts_to_sequences(captions)

    # 6. Find the maximum length of any caption in our dataset (including start/end tokens)
    max_len = max(len(s) for s in train_seqs)

    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Max caption length: {max_len}")

    return tokenizer, max_len
