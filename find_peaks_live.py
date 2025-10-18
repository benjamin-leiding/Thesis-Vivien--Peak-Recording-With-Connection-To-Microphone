import pyaudio
import numpy as np
import wave
from collections import deque
import time
import pandas as pd
import os

SAMPLING_RATE = 384000
PRESAVE_TIME_MS = 1
POSTSAVE_TIME_MS = 1
TRESHOLDS = [2000, 1500, 2500]
CHANNELS = 1
CHUNK_SIZE = 4096
TIME_TO_CHECK_SEC = 10

# Convert ms to samples
pre_samples = int(SAMPLING_RATE * PRESAVE_TIME_MS / 1000)
post_samples = int(SAMPLING_RATE * POSTSAVE_TIME_MS / 1000)

# Circular buffer to store pre-peak samples
pre_buffer = deque(maxlen=pre_samples)

# PyAudio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLING_RATE,
                input=True,
                input_device_index=1, #personal value, from file get_micro_info.py
                frames_per_buffer=CHUNK_SIZE)

print("Listening for peaks...")

peak_log = []
start_time = time.time()
peak_count = 0

try:
    for t in TRESHOLDS:
        directory = f"records_threshold_{t}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        peak_count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < TIME_TO_CHECK_SEC:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)

            for s in samples:
                pre_buffer.append(s)
                sample = abs(s)
                if  sample > t: 
                    # Peak detected
                    peak_time = time.time() - start_time
                    peak_amplitude = float(sample)
                    
                    post_buffer = [s]
                    
                    # Read post_samples from stream
                    while len(post_buffer) < post_samples:
                        data_post = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        post_chunk = np.frombuffer(data_post, dtype=np.int16)
                        post_buffer.extend(post_chunk)

                    # Combine pre + post
                    peak_window = list(pre_buffer) + post_buffer[:post_samples]

                    # Save WAV
                    peak_count += 1
                    filename = f"/peak_capture_{peak_count}_by_threshold_{t}.wav"
                    wf = wave.open(directory + filename, 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(SAMPLING_RATE)
                    wf.writeframes(np.array(peak_window, dtype=np.int16).tobytes())
                    wf.close()
                    
                    signal = np.array(peak_window, dtype=np.int16)
                    fft_spectrum = np.fft.fft(signal)
                    frequencies = np.fft.fftfreq(len(fft_spectrum), d=1/SAMPLING_RATE)
                    fft_magnitude = np.abs(fft_spectrum[:len(frequencies)//2])
                    frequencies = frequencies[:len(frequencies)//2]
                    dominant_freq = frequencies[np.argmax(fft_magnitude)]

                    # Cache metadata
                    peak_log.append({
                        "time": peak_time,
                        "threshold": t,
                        "frequencyHz": dominant_freq,
                        "amplitude": peak_amplitude,
                        "filename": filename
                    })

                    print(f"[{time.strftime('%H:%M:%S')}] Peak No. {peak_count} | "
                        f"Threshold={t} | FrequencyHz={dominant_freq:.1f} Hz | "
                        f"Amplitude={peak_amplitude:.1f} | Filename={filename}")

                    # Clear pre-buffer for next detection
                    pre_buffer.clear()
                    
except KeyboardInterrupt:
    print("Stopping...")

finally:
    df = pd.DataFrame(peak_log)
    df.to_csv("peak_log.csv", index=False)
    stream.stop_stream()
    stream.close()
    p.terminate()
