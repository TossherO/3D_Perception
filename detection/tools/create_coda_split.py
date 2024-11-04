import os
import random
import json

label_path = './data/CODA/3d_bbox/os1'
label_dirs = os.listdir(label_path)

scenes = []
for label_dir in label_dirs:
    label_files = os.listdir(os.path.join(label_path, label_dir))
    frames = []
    for label_file in label_files:
        frames.append(int(label_file.split('.')[0].split('_')[-1]))
    if len(frames) == 0:
        continue
    frames = sorted(frames)
    
    sequences = []
    sequence = [frames[0]]
    for i in range(1, len(frames)):
        if frames[i] - frames[i-1] == 1:
            sequence.append(frames[i])
        else:
            sequences.append(sequence)
            sequence = [frames[i]]
    sequences.append(sequence)
    
    for sequence in sequences:
        scenes.append({
            'scene': label_dir,
            'frames': sequence,
            'frame_len': len(sequence)
        })
        
random.shuffle(scenes)
cusum = [0]
for scene in scenes:
    cusum.append(cusum[-1] + scene['frame_len'])
percents = [item / cusum[-1] for item in cusum]

train_val_split = 0.5
train_val_split_idx = 0
for i in range(len(percents)):
    if percents[i] > train_val_split:
        train_val_split_idx = i
        break
train_scenes = scenes[:train_val_split_idx]
val_scenes = scenes[train_val_split_idx:]
train_info = {
    'scenes': train_scenes,
    'scene_len': len(train_scenes),
    'frame_len': cusum[train_val_split_idx]
}
val_info = {
    'scenes': val_scenes,
    'scene_len': len(val_scenes),
    'frame_len': cusum[-1] - cusum[train_val_split_idx]
}

save_file = './data/CODA/split.json'
with open(save_file, 'w') as f:
    json.dump({
        'train': train_info,
        'val': val_info
    }, f, indent=4)