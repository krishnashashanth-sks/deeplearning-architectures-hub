import numpy as np

def inference_on_mel_spectrogram(model, mel_spectrogram_input, labels_df, top_n=5):
    """
    Performs inference on a single preprocessed Mel-spectrogram and returns top predicted classes.

    Args:
        model (tf.keras.Model): The trained YAMNet model.
        mel_spectrogram_input (tf.Tensor): A single preprocessed Mel-spectrogram
                                          with shape (1, time_steps, features, channels).
        labels_df (pd.DataFrame): DataFrame containing the class labels with 'index' and 'display_name' columns.
        top_n (int): Number of top predictions to return.

    Returns:
        list: A list of tuples, each containing (class_name, probability).
    """
    if mel_spectrogram_input.ndim != 4 or mel_spectrogram_input.shape[0] != 1:
        raise ValueError("Input mel_spectrogram_input must have shape (1, time_steps, features, channels).")

    predictions = model.predict(mel_spectrogram_input)
    # Assuming predictions are probabilities (sigmoid activated)
    probabilities = predictions[0]

    # Get the indices of the top N probabilities
    top_n_indices = np.argsort(probabilities)[::-1][:top_n]

    results = []
    for i in top_n_indices:
        class_name = labels_df.loc[labels_df['index'] == i, 'display_name'].iloc[0]
        probability = probabilities[i]
        results.append((class_name, probability))

    return results