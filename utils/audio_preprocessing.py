import librosa
import numpy as np


def compute_spectrogram(audio, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    """Compute spectrogram from audio waveform.
    :param audio: audio waveform
    :param sr: sampling rate
    :param n_fft: length of FFT window
    :param hop_length: number of samples between successive frames
    :param n_mels: number of Mel bands to generate
    :return: spectrogram"""
    # Compute spectrogram
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    logspec = librosa.power_to_db(spec, ref=np.max)
    logspec = logspec.transpose()

    # Normalize spectrogram
    mean = np.mean(logspec)
    std = np.std(logspec)

    return (logspec - mean) / std


def spectrogram_to_audio(mel_spec, sr=22050, n_fft=1024, hop_length=256, win_length=1024):
    """Convert spectrogram to audio waveform.
    :param mel_spec: spectrogram
    :param sr: sampling rate
    :param n_fft: length of FFT window
    :param hop_length: number of samples between successive frames
    :param win_length: window length
    :return: audio waveform"""
    # Convert mel-spectrogram to linear spectrogram
    mel_spec = mel_spec.transpose()
    linear_spec = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr, n_fft=n_fft)

    # Convert linear spectrogram to audio waveform
    return librosa.core.istft(
        linear_spec,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
    )
