import librosa
import numpy as np


def calculate_audio_features(y, sr):
    # Calculate Chroma Short-Time Fourier Transform (STFT)
    chroma_stft_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    chroma_stft_var = np.var(librosa.feature.chroma_stft(y=y, sr=sr))

    # Calculate Root Mean Square (RMS)
    rms_mean = np.mean(librosa.feature.rms(y=y))
    rms_var = np.var(librosa.feature.rms(y=y))

    # Calculate Spectral Centroid
    spectral_centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_centroid_var = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Calculate Spectral Bandwidth
    spectral_bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_bandwidth_var = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Calculate Rolloff
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rolloff_var = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Calculate Zero Crossing Rate
    zero_crossing_rate_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    zero_crossing_rate_var = np.var(librosa.feature.zero_crossing_rate(y=y))

    # Calculate Harmony
    harmony = librosa.effects.harmonic(y=y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)

    # Calculate Perceptr
    perceptr = librosa.effects.percussive(y=y)
    perceptr_mean = np.mean(perceptr)
    perceptr_var = np.var(perceptr)

    # Calculate Tempo
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]

    # Calculate MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Calculate 20 MFCCs
    mfcc_means = np.mean(mfccs, axis=1)  # Calculate means of MFCCs
    mfcc_vars = np.var(mfccs, axis=1)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Calculate mean and variance for each MFCC coefficient
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_variances = np.var(mfccs, axis=1)

    # Assign values to individual variables
    mfcc1_mean, mfcc2_mean, mfcc3_mean, mfcc4_mean, mfcc5_mean, \
    mfcc6_mean, mfcc7_mean, mfcc8_mean, mfcc9_mean, mfcc10_mean, \
    mfcc11_mean, mfcc12_mean, mfcc13_mean, mfcc14_mean, mfcc15_mean, \
    mfcc16_mean, mfcc17_mean, mfcc18_mean, mfcc19_mean, mfcc20_mean = mfcc_means

    mfcc1_var, mfcc2_var, mfcc3_var, mfcc4_var, mfcc5_var, \
    mfcc6_var, mfcc7_var, mfcc8_var, mfcc9_var, mfcc10_var, \
    mfcc11_var, mfcc12_var, mfcc13_var, mfcc14_var, mfcc15_var, \
    mfcc16_var, mfcc17_var, mfcc18_var, mfcc19_var, mfcc20_var = mfcc_variances


	all_features = [chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
					spectral_centroid_mean, spectral_centroid_var,
					spectral_bandwidth_mean, spectral_bandwidth_var,
					rolloff_mean, rolloff_var,
					zero_crossing_rate_mean, zero_crossing_rate_var,
					harmony_mean, harmony_var,
					perceptr_mean, perceptr_var, tempo,
					mfcc1_mean, mfcc1_var,
					mfcc2_mean, mfcc2_var,
					mfcc3_mean, mfcc3_var,
					mfcc4_mean, mfcc4_var,
					mfcc5_mean, mfcc5_var,
					mfcc6_mean, mfcc6_var,
					mfcc7_mean, mfcc7_var,
					mfcc8_mean, mfcc8_var,
					mfcc9_mean, mfcc9_var,
					mfcc10_mean, mfcc10_var,
					mfcc11_mean, mfcc11_var,
					mfcc12_mean, mfcc12_var,
					mfcc13_mean, mfcc13_var,
					mfcc14_mean, mfcc14_var,
					mfcc15_mean, mfcc15_var,
					mfcc16_mean, mfcc16_var,
					mfcc17_mean, mfcc17_var,
					mfcc18_mean, mfcc18_var,
					mfcc19_mean, mfcc19_var,
					mfcc20_mean, mfcc20_var]

	combined_features = np.array(all_features)
	
	return combined_features



def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    return calculate_audio_features(y, sr)