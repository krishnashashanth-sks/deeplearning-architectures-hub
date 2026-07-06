def predict_multistream_cnn(multistream_cnn_model,stream1_data, stream2_data):
    """
    Makes predictions using the multistream CNN model.

    Args:
        stream1_data (numpy.ndarray or tf.Tensor): Preprocessed input data for stream 1.
        stream2_data (numpy.ndarray or tf.Tensor): Preprocessed input data for stream 2.

    Returns:
        numpy.ndarray: The prediction probabilities from the model.
    """
    predictions = multistream_cnn_model.predict([stream1_data, stream2_data])
    return predictions
