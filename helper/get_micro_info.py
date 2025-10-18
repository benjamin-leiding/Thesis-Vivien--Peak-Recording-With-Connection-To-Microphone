import pyaudio

p = pyaudio.PyAudio()
print("Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"{i}: {info['name']}")

DEVICE_INDEX = 1 # individual, use i from above

info = p.get_device_info_by_index(DEVICE_INDEX)
sampling_rate = info['defaultSampleRate']
print(f"supported sampling rate: {sampling_rate}") 
if sampling_rate != 384000:
    print("! please make sure, you use the correct input device")
    
p.terminate()
