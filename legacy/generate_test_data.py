import os
import numpy as np
import soundfile as sf

def generate_kick(path, sr=44100, duration=0.5):
    t = np.linspace(0, duration, int(sr * duration))
    # Frequency sweep for kick
    freq = 150 * np.exp(-15 * t) + 40
    y = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    # Envelope
    env = np.exp(-10 * t)
    y = y * env
    sf.write(path, y, sr)

def generate_snare(path, sr=44100, duration=0.5):
    t = np.linspace(0, duration, int(sr * duration))
    # Noise + tone
    noise = np.random.normal(0, 0.1, len(t))
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-20 * t)
    y = (noise + tone) * np.exp(-15 * t)
    sf.write(path, y, sr)

def generate_hat(path, sr=44100, duration=0.1):
    t = np.linspace(0, duration, int(sr * duration))
    # High pass noise
    noise = np.random.normal(0, 0.1, len(t))
    # Simple high pass filter simulation
    y = noise * np.exp(-50 * t)
    sf.write(path, y, sr)

def create_test_dataset(base_dir="test_samples"):
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate 5 kicks
    for i in range(5):
        generate_kick(os.path.join(base_dir, f"kick_{i}.wav"))
    
    # Generate 5 snares
    for i in range(5):
        generate_snare(os.path.join(base_dir, f"snare_{i}.wav"))
    
    # Generate 5 hats
    for i in range(5):
        generate_hat(os.path.join(base_dir, f"hat_{i}.wav"))

    print(f"Test dataset created in {base_dir}")

if __name__ == "__main__":
    create_test_dataset()
