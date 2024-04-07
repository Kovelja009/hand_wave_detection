import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from scipy.signal import butter, filtfilt

def highpass_filter(signal, cutoff_frequency=0.1, sampling_rate=30):
    # Design a Butterworth high-pass filter
    order = 4  # Filter order
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


''' returns a dictionary of features extracted from the x and y signals '''
def calculate_features(x_signal, y_signal, sampling_rate=30, cutoff_frequency = 0.1, width=1920, height=1080): 
    x_signal = np.array(x_signal)
    y_signal = np.array(y_signal)
  
    # signal min max scaling
    x_signal = x_signal/width
    y_signal = y_signal/height

    # Amplitude-related features
    max_amplitude_x = np.max(x_signal)
    max_amplitude_y = np.max(y_signal)

    min_amplitude_x = np.min(x_signal)
    min_amplitude_y = np.min(y_signal)

    avg_amplitude_x = np.mean(x_signal)
    avg_amplitude_y = np.mean(y_signal)

    rms_amplitude_x = np.sqrt(np.mean(x_signal**2))
    rms_amplitude_y = np.sqrt(np.mean(y_signal**2))

    # Temporal features
    rise_time_x = 0.1 * (np.argmax(x_signal > 0.1 * max_amplitude_x) - np.argmax(x_signal > 0.9 * max_amplitude_x)) / sampling_rate
    rise_time_y = 0.1 * (np.argmax(y_signal > 0.1 * max_amplitude_y) - np.argmax(y_signal > 0.9 * max_amplitude_y)) / sampling_rate

    fall_time_x = 0.1 * (np.argmax(x_signal > 0.9 * max_amplitude_x) - np.argmax(x_signal > 0.1 * max_amplitude_x)) / sampling_rate
    fall_time_y = 0.1 * (np.argmax(y_signal > 0.9 * max_amplitude_y) - np.argmax(y_signal > 0.1 * max_amplitude_y)) / sampling_rate

    peak_to_peak_time_x = (np.argmax(x_signal > 0.9 * max_amplitude_x) - np.argmax(x_signal > 0.1 * max_amplitude_x)) / sampling_rate
    peak_to_peak_time_y = (np.argmax(y_signal > 0.9 * max_amplitude_y) - np.argmax(y_signal > 0.1 * max_amplitude_y)) / sampling_rate

    # Nonlinear features
    x_signal_kurtosis = kurtosis(x_signal)
    y_signal_kurtosis = kurtosis(y_signal)

    x_signal_skewness = skew(x_signal)
    y_signal_skewness = skew(y_signal)

    # Zero-crossing features with filtered signal
    filtered_x_signal = highpass_filter(x_signal, cutoff_frequency=cutoff_frequency, sampling_rate=sampling_rate)
    filtered_y_signal = highpass_filter(y_signal, cutoff_frequency=cutoff_frequency, sampling_rate=sampling_rate)

    zero_crossing_rate_x = np.sum(np.diff(np.sign(filtered_x_signal)) != 0) / (2 * len(filtered_x_signal) / sampling_rate)
    zero_crossing_rate_y = np.sum(np.diff(np.sign(filtered_y_signal)) != 0) / (2 * len(filtered_y_signal) / sampling_rate)

    # FFT
    fft_x_signal = fft(x_signal)
    fft_y_signal = fft(y_signal)

    fft3_x = np.abs(fft_x_signal[2])
    fft3_y = np.abs(fft_y_signal[2])

    # Peaks
    peaks_x, _ = find_peaks(x_signal)
    peaks_y, _ = find_peaks(y_signal)

    num_peaks_x = len(peaks_x)
    num_peaks_y = len(peaks_y)

    return {
        'x_max_amplitude': max_amplitude_x,
        'y_max_amplitude': max_amplitude_y,
        'x_min_amplitude': min_amplitude_x,
        'y_min_amplitude': min_amplitude_y,
        'x_avg_amplitude': avg_amplitude_x,
        'y_avg_amplitude': avg_amplitude_y,
        'x_rms_amplitude': rms_amplitude_x,
        'y_rms_amplitude': rms_amplitude_y,
        'x_rise_time': rise_time_x,
        'y_rise_time': rise_time_y,
        'x_fall_time': fall_time_x,
        'y_fall_time': fall_time_y,
        'x_peak_to_peak_time': peak_to_peak_time_x,
        'y_peak_to_peak_time': peak_to_peak_time_y,
        'x_signal_kurtosis': x_signal_kurtosis,
        'y_signal_kurtosis': y_signal_kurtosis,
        'x_signal_skewness': x_signal_skewness,
        'y_signal_skewness': y_signal_skewness,
        'x_zero_crossing_rate': zero_crossing_rate_x,
        'y_zero_crossing_rate': zero_crossing_rate_y,
        'x_fft3': fft3_x,
        'y_fft3': fft3_y,
        'x_num_peaks': num_peaks_x,
        'y_num_peaks': num_peaks_y
    }