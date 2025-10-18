import time
import numpy as np
import pyaudio


SAMPLING_RATE = 384000
CHANNEL = 1
DEVICE_INDEX = 1 # individual with get_micro_info.py
FRAMES_PER_BUFFER = 4096

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLING_RATE,
                input=True,
                input_device_index=DEVICE_INDEX,  # optional
                frames_per_buffer=FRAMES_PER_BUFFER)

try:
    while True:
        # Read chunk from mic
        data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        signal = np.frombuffer(data, dtype=np.int16)

        # FFT
        fft_spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(fft_spectrum), d=1/SAMPLING_RATE)

        # Positive frequencies only
        magnitude = np.abs(fft_spectrum[:len(freqs)//2])
        freqs = freqs[:len(freqs)//2]

        # Dominant frequency
        dominant_freq = freqs[np.argmax(magnitude)]

        # Log
        print(f"[{time.strftime('%H:%M:%S')}] Frequency: {dominant_freq:.1f} Hz")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()