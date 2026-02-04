#generate MLAAD & MLAAIO dataset via recursive access

import os
import json
import random

# Initialize empty list to store all data entries
data = []

# Function to create json entry for a file
def create_json_entry(filepath, is_spoof):
    label = "spoof" if is_spoof else "bonafide"
    return {
        "wav": filepath,
        "question": "Is this audio bonafide or spoofed? \n bonafide\n spoofed",
        "answer": label,
        "mapped_label": label
    }

# Process MLAAD (spoofed) data
base_dir = "/data/sls/scratch/mvideet/datasets/MLAAD/fake/en"
dir_list = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
for model in dir_list:
    model_dir = os.path.join(base_dir, model)
    all_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    for file in all_files:
        if file.endswith(".wav"):
            data.append(create_json_entry(os.path.join(model_dir, file), True))

# Process MLAAIO (real) data
base_dir = "/data/sls/scratch/mvideet/datasets/en_US/by_book"
voices = ["male", "female"]
for voice in voices:
    voice_dir = os.path.join(base_dir, voice)
    # Get list of all speakers (book directories) for this voice type
    speaker_list = [d for d in os.listdir(voice_dir) if os.path.isdir(os.path.join(voice_dir, d))]
    for speaker in speaker_list:
        speaker_dir = os.path.join(voice_dir, speaker)
        # Get list of all stories (subdirectories) for this speaker
        story_list = [d for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir, d))]
        for story in story_list:
            story_dir = os.path.join(speaker_dir, story)
            # Go into the wav folder within each story directory
            wav_dir = os.path.join(story_dir, "wavs")
            if os.path.isdir(wav_dir):
                # Get all wav files in this directory, excluding macOS metadata files
                wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav") and not f.startswith("._")]
                for wav_file in wav_files:
                    data.append(create_json_entry(os.path.join(wav_dir, wav_file), False))

# Shuffle the data for random split
random.shuffle(data)

# Calculate split indices
total_samples = len(data)
train_size = int(0.8 * total_samples)

# Split the data
train_data = data[:train_size]
test_data = data[train_size:]

print(f"Total samples: {total_samples}")
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Save train and test splits
with open("/data/sls/u/urop/mvideet/sparse_audio/dataset_processing/MLAAD_MLAAIO_train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("/data/sls/u/urop/mvideet/sparse_audio/dataset_processing/MLAAD_MLAAIO_test.json", "w") as f:
    json.dump(test_data, f, indent=4)

print("Saved train and test splits successfully!")