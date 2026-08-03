import os
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Supported audio extensions
AUDIO_EXTENSIONS = ('.wav', '.aif', '.aiff', '.mp3')

def get_audio_files(directory):
    """Recursively find all audio files in the target directory."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(os.path.join(root, file))
    return audio_files

def extract_features(file_path):
    """Extract refined audio features using a dynamic 30% window and texture tracking."""
    try:
        # 1. Load the full file first to trim and calculate percentage
        # (We use librosa.load with sr=None to keep native sample rate)
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr

        if len(y) == 0:
            return None

        # 2. Trim silence from both ends
        y_trimmed, index = librosa.effects.trim(y, top_db=30)
        
        # 3. Calculate 50% window of the active sound
        active_length = len(y_trimmed)
        analysis_length = int(active_length * 1)
        
        # Ensure we have at least some data to analyze
        if analysis_length < 512:
             analysis_length = min(active_length, 2048)
             
        y_active = y_trimmed[:analysis_length]

        # 4. Extract Features
        # MFCCs (Timbre) - Mean and Std Dev to capture texture movement
        mfcc = librosa.feature.mfcc(y=y_active, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Spectral Centroid (Brightness) - Mean and Std Dev
        centroid = librosa.feature.spectral_centroid(y=y_active, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)

        # Zero Crossing Rate (Noise vs Tonal)
        zcr = librosa.feature.zero_crossing_rate(y=y_active)
        zcr_mean = np.mean(zcr)

        # Spectral Rolloff (High-frequency shape)
        rolloff = librosa.feature.spectral_rolloff(y=y_active, sr=sr)
        rolloff_mean = np.mean(rolloff)

        # Log-Attack Time (Transient sharpness)
        onset_env = librosa.onset.onset_strength(y=y_active, sr=sr)
        if len(onset_env) > 0:
            peak_idx = np.argmax(onset_env)
            frames = np.arange(len(onset_env))
            times = librosa.frames_to_time(frames, sr=sr)
            attack_time = times[peak_idx]
            log_attack_time = np.log10(max(attack_time, 0.0001))
        else:
            log_attack_time = -4.0

        # RMS Energy
        rms = librosa.feature.rms(y=y_active)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Spectral Flatness (Tonal vs Noise)
        flatness = librosa.feature.spectral_flatness(y=y_active)
        flatness_mean = np.mean(flatness)
        
        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y_active, sr=sr)
        bandwidth_mean = np.mean(bandwidth)

        # Spectral Contrast (maps harmonic vs percussive peaks)
        contrast = librosa.feature.spectral_contrast(y=y_active, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)

        # Harmonic-to-Percussive Ratio (HPR)
        harmonic, percussive = librosa.effects.hpss(y_active)
        h_rms = np.mean(librosa.feature.rms(y=harmonic))
        p_rms = np.mean(librosa.feature.rms(y=percussive))
        hpr = h_rms / (p_rms + 1e-6)

        # Sub-Bass Energy Ratio (Energy below 60Hz)
        stft = np.abs(librosa.stft(y_active))
        freqs = librosa.fft_frequencies(sr=sr)
        sub_bass_mask = freqs <= 60
        sub_bass_energy = np.sum(stft[sub_bass_mask, :])
        total_energy = np.sum(stft)
        sub_bass_ratio = sub_bass_energy / (total_energy + 1e-6)

        # Effective Duration & Energy Center (Concentration of sound)
        # We analyze the full signal to capture fading and trailing silence accurately
        rms_full = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        threshold_rms = np.max(rms_full) * librosa.db_to_amplitude(-40)
        above_threshold = np.where(rms_full > threshold_rms)[0]
        if len(above_threshold) > 0:
            effective_duration = librosa.frames_to_time(above_threshold[-1], sr=sr, hop_length=512)
        else:
            effective_duration = duration
        
        times_full = librosa.frames_to_time(np.arange(len(rms_full)), sr=sr, hop_length=512)
        energy_center = np.sum(times_full * rms_full) / (np.sum(rms_full) + 1e-6)

        # Crest Factor (Peakiness)
        peak_amp = np.max(np.abs(y_active))
        crest_factor = peak_amp / (rms_mean + 1e-6)

        # Transient-to-Sustain Ratio (First 50ms vs rest)
        transient_samples = int(0.05 * sr)
        transient_energy = np.sum(y_active[:transient_samples]**2)
        sustain_energy = np.sum(y_active[transient_samples:]**2)
        transient_ratio = transient_energy / (sustain_energy + 1e-6)

        # Feature vector construction:
        feature_vector = np.concatenate([
            mfcc_mean,
            mfcc_std,
            contrast_mean,
            [
                centroid_mean, centroid_std, zcr_mean, rolloff_mean, 
                log_attack_time, effective_duration, rms_mean, rms_std, 
                flatness_mean, bandwidth_mean, hpr, sub_bass_ratio,
                energy_center, crest_factor, transient_ratio
            ]
        ])

        return {
            "path": file_path,
            "filename": os.path.basename(file_path),
            "duration": duration,
            "effective_duration": effective_duration,
            "features": feature_vector.tolist()
        }
    except Exception as e:
        # Silently fail for individual files to keep the pipeline moving
        return None

def process_pipeline(input_dir, output_file):
    print(f"--- PulseMap Pipeline Starting (Refined Mode) ---")
    start_time = time.time()

    print(f"Scanning directory: {input_dir}")
    audio_files = get_audio_files(input_dir)
    print(f"Found {len(audio_files)} audio files.")

    if not audio_files:
        print("No audio files found. Exiting.")
        return

    print(f"Extracting features using {mp.cpu_count()} cores...")
    print(f"(Trimming silence + Analyzing active sound)")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(extract_features, audio_files)
    
    valid_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_results)} files.")

    if not valid_results:
        print("No valid features extracted. Exiting.")
        return

    print("Running UMAP and HDBSCAN...")
    feature_matrix = np.array([r['features'] for r in valid_results])
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Manual weighting: Increase importance of specific musical features
    # Index 33: Spectral Centroid (Average Frequency)
    scaled_features[:, 33] *= 2.0
    # Index 38: Effective Duration (ignores silent tails)
    scaled_features[:, 38] *= 2.5
    # Index 43: Harmonic-to-Percussive Ratio (Distinguishes Bass from Kick)
    scaled_features[:, 43] *= 3.0
    # Index 45: Energy Center (Concentration of sound)
    scaled_features[:, 45] *= 2.5
    # Index 46: Crest Factor (Peakiness)
    scaled_features[:, 46] *= 2.5
    # Index 47: Transient Ratio (Initial hit strength)
    scaled_features[:, 47] *= 2.0

    # UMAP: Lower n_neighbors (15 -> 7) to prioritize local uniqueness (better for glitches/outliers)
    reducer = umap.UMAP(
        n_neighbors=2, 
        min_dist=0.1, 
        n_components=2, 
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(scaled_features)

    # HDBSCAN: Tuning for more conservative clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=3,           # Higher value makes clustering more selective
        gen_min_span_tree=True
    )
    cluster_labels = clusterer.fit_predict(embedding)
    outlier_scores = clusterer.outlier_scores_

    # POST-ANALYSIS REFINEMENT
    # 1. Purify clusters based on high-dimensional feature similarity
    next_cluster_id = int(np.max(cluster_labels)) + 1
    refined_labels = cluster_labels.copy()
    similarity_scores = np.ones(len(cluster_labels)) # Default to 1.0 (perfect match)
    
    unique_base_labels = set(cluster_labels)
    for label in unique_base_labels:
        if label == -1: continue
        
        # Get indices and features for this cluster
        indices = np.where(cluster_labels == label)[0]
        cluster_features = scaled_features[indices]
        
        # Calculate high-dimensional centroid (the "average sound")
        hi_dim_centroid = cluster_features.mean(axis=0)
        
        # Calculate Euclidean distances of all members to this centroid
        distances = np.linalg.norm(cluster_features - hi_dim_centroid, axis=1)
        
        if len(distances) > 1:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            # Normalize distances to 0.0 - 1.0 similarity score (1.0 is at centroid, 0.0 is very far)
            # Use 2 * std_dist as the "maximum" distance for 0 similarity
            max_range = std_dist * 2 + 1e-6
            for i, idx in enumerate(indices):
                score = max(0.0, 1.0 - (distances[i] / max_range))
                similarity_scores[idx] = score

            # Threshold: Any file > 1.0 standard deviations from the cluster average is evicted
            threshold = mean_dist + (1.0 * std_dist)
            for i, idx in enumerate(indices):
                if distances[i] > threshold:
                    refined_labels[idx] = next_cluster_id
                    next_cluster_id += 1
        else:
            # Single-item base clusters get 0.5 similarity by default
            similarity_scores[indices[0]] = 0.5
        
    # 2. Assign unique IDs to remaining generic noise points
    for i in range(len(refined_labels)):
        if refined_labels[i] == -1:
            refined_labels[i] = next_cluster_id
            next_cluster_id += 1
            similarity_scores[i] = 0.2 # Noise is inherently low-gravity

    # 3. Calculate final 2D centroids for the visualization
    centroids_2d = {}
    unique_refined_labels = set(refined_labels)
    for label in unique_refined_labels:
        mask = (refined_labels == label)
        cluster_points_2d = embedding[mask]
        centroids_2d[label] = cluster_points_2d.mean(axis=0)

    final_data = []
    for i, res in enumerate(valid_results):
        label = int(refined_labels[i])
        x_coord, y_coord = centroids_2d[label]

        final_data.append({
            "filename": res['filename'],
            "path": res['path'],
            "duration": res['duration'],
            "effective_duration": float(res['effective_duration']),
            "x": float(x_coord),
            "y": float(y_coord),
            "cluster": label,
            "uniqueness": float(outlier_scores[i]),
            "gravity": float(similarity_scores[i])
        })

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    end_time = time.time()
    print(f"--- Pipeline Finished in {end_time - start_time:.2f}s ---")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PulseMap Audio Data Pipeline")
    parser.add_argument("input_dir", help="Directory containing audio files")
    parser.add_argument("--output", default="data.json", help="Output JSON file path")
    
    args = parser.parse_args()
    process_pipeline(args.input_dir, args.output)
