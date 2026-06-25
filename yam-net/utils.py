import tensorflow as tf
from yt_dlp import YoutubeDL
import pandas as pd
import os

# Configuration parameters (these should match YAMNet's original config as closely as possible)
SAMPLE_RATE = 16000
STFT_WINDOW_SECONDS = 0.025  # 25ms
STFT_HOP_SECONDS = 0.010     # 10ms
MEL_BANDS = 64
AUDIO_DURATION = 0.96      # YAMNet processes 0.96s clips

def waveform_to_log_mel_spectrogram(waveform, sample_rate):
    # Ensure waveform is float32
    waveform = tf.cast(waveform, tf.float32)

    # Calculate frame length and hop length in samples
    frame_length = int(round(sample_rate * STFT_WINDOW_SECONDS))
    frame_step = int(round(sample_rate * STFT_HOP_SECONDS))

    # Pad the waveform if it's shorter than the required duration
    target_length = int(sample_rate * AUDIO_DURATION)
    waveform = tf.pad(waveform, [[0, tf.maximum(0, target_length - tf.shape(waveform)[0])]])
    waveform = waveform[:target_length] # Trim if too long

    # Compute STFT
    stft = tf.signal.stft(
        waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=frame_length,
        window_fn=tf.signal.hann_window,
        pad_end=False
    )
    spectrogram = tf.abs(stft)

    # Create a Mel filterbank
    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=MEL_BANDS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0
    )

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute log mel spectrogram
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    return log_mel_spectrogram

def download_audioset_clip(youtube_id, start_time, end_time, output_dir='audio_clips'):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/{youtube_id}_{start_time}-{end_time}.%(ext)s', # Added start/end to filename
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ss', str(start_time),
            '-to', str(end_time)
        ],
        'prefer_ffmpeg': True,
        'quiet': True, # Suppress yt_dlp output
        'no_warnings': True,
        'retries': 3 # Retry failed downloads
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={youtube_id}']) # Removed extra ')'
        print(f"Downloaded {youtube_id} from {start_time} to {end_time}")
    except Exception as e:
        print(f"Failed to download {youtube_id} from {start_time} to {end_time}: {e}")