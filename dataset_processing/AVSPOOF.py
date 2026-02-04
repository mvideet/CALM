import json

with open("/data/sls/scratch/mvideet/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "r") as f:
    lines = f.readlines()
import random

data = []
question = "Is this audio bonafide or spoofed? \n bonafide\n spoofed"

base_dir = "/data/sls/scratch/mvideet/datasets/LA/ASVspoof2019_LA_eval/flac"
for line in lines:
    speaker_id, utt_id, _, attack_id, label = line.strip().split(" ")
    new_file_loc = f"{base_dir}/{utt_id}.flac"
    
    # Set answer based on original label
    if label == 'bonafide':
        label = 'bonafide'
        mapped = "bonafide"
    else:  # spoof
        label = 'spoof' 
        mapped = "spoof"
            
    entry = {
        'wav': new_file_loc,
        'question': question,
        'answer': label,
        'mapped_label': mapped
    }
    data.append(entry)


# Create a smaller dataset for sanity checking
# Take 100 samples randomly from the full dataset
sanity_data = random.sample(data, min(20, len(data)))

# # Save the sanity check dataset
with open("LA_eval_sanity_20.json", "w") as f:
    json.dump(sanity_data, f, indent=4)

# with open("LA_train.json", "w") as f:
#     json.dump(data, f, indent=4)