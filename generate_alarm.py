"""
Generates a short alarm WAV file for the drowsiness detection system.
Run this script once to create alarm.wav before using main.py.
"""
import wave
import struct
import math
import os

def generate_alarm(filename="alarm.wav", duration=2.0, sample_rate=44100):
    """Generate a pulsing two-tone alarm sound."""
    n_samples = int(sample_rate * duration)
    samples = []

    freq1 = 880   # A5 - high urgency tone
    freq2 = 1100  # ~C#6 - alternating tone
    pulse_rate = 6  # pulses per second

    for i in range(n_samples):
        t = i / sample_rate
        # Alternate between two frequencies for urgency
        freq = freq1 if int(t * pulse_rate) % 2 == 0 else freq2
        # Pulsing envelope
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * pulse_rate * t)
        # Generate the sample
        sample = envelope * math.sin(2 * math.pi * freq * t)
        # Convert to 16-bit PCM
        samples.append(int(sample * 32000))

    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)
        for s in samples:
            wav_file.writeframes(struct.pack('<h', s))

    print(f"Alarm sound generated: {filename}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alarm_path = os.path.join(script_dir, "alarm.wav")
    generate_alarm(alarm_path)
