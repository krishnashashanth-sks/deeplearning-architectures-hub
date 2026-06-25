import matplotlib.pyplot as plt
import tensorflow as tf
from utils import *
import librosa
from model import build_yamnet_model
from dataset import X_train,y_train,X_test,y_test,X_val,y_val,BATCH_SIZE,labels_df
from inference import inference_on_mel_spectrogram
import numpy as np

# Replace 'path/to/your/audio.wav' with an actual audio file
# audio, sr = librosa.load('path/to/your/audio.wav', sr=SAMPLE_RATE, mono=True)
# You would load batch of audio clips here.

# For demonstration, let's create a dummy waveform
dummy_waveform = tf.random.uniform(shape=[int(SAMPLE_RATE * AUDIO_DURATION)], minval=-0.5, maxval=0.5)
log_melspec = waveform_to_log_mel_spectrogram(dummy_waveform, SAMPLE_RATE)

print(f"Log Mel Spectrogram shape: {log_melspec.shape}")

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_melspec.numpy().T,
                         sr=SAMPLE_RATE,
                         x_axis='time',
                         y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Log Mel Spectrogram Example')
plt.tight_layout()
plt.show()

custom_yamnet_model = build_yamnet_model(input_shape=(log_melspec.shape[0], log_melspec.shape[1], 1))

custom_yamnet_model.summary()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False) # Use from_logits=False if sigmoid activation is on output layer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(multi_label=True)]

custom_yamnet_model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

custom_yamnet_model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=1, # Start with a few epochs, then increase
    validation_data=(X_val, y_val)
)

loss, binary_accuracy, auc = custom_yamnet_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Binary Accuracy: {binary_accuracy}")
print(f"Test AUC: {auc}")

print("\n--- Inference Example ---")

# Select the first example from the test set for inference
single_test_example = X_test[0:1] # Keep batch dimension

# Perform inference
top_predictions = inference_on_mel_spectrogram(custom_yamnet_model, single_test_example, labels_df, top_n=5)

print(f"Top {len(top_predictions)} predictions for the first test example:")
for class_name, prob in top_predictions:
    print(f"- {class_name}: {prob:.4f}")

# You can also get the true labels for comparison if y_test is available
if 'y_test' in locals() and len(y_test) > 0:
    true_labels_for_example = y_test[0].numpy()
    true_label_indices = np.where(true_labels_for_example == 1)[0]
    if len(true_label_indices) > 0:
        print("\nTrue labels for this example:")
        for true_idx in true_label_indices:
            true_class_name = labels_df.loc[labels_df['index'] == true_idx, 'display_name'].iloc[0]
            print(f"- {true_class_name}")
    else:
        print("\nNo true labels assigned to this example in the dummy data.")