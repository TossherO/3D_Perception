import os
import random
import json

# | Sequence ID   | Person    | Trajectory     | Time (secs) | Traffic Level | Weather/Brightness| Date       |
# | ------------- | --------- | ------------   | ----------- | ------------- | ----------------- | ---------- |
# | 0             | Raymond   | GDC (F)        | 821.2       | Light         | Cloudy            | 2023-01-16 |
# | 1             | Arthur    | GDC (B)        | 870.4       | Light         | Cloudy            | 2023-01-16 |
# | 2             | Chaitanya | Guad (F)       | 1669.3      | Light         | Dark              | 2023-01-16 |
# | 3             | Chaitanya | GDC (B)        | 1472.3      | Medium        | Sunny             | 2023-01-17 |
# | 4             | Arthur    | GDC (B)        | 823.6       | Light         | Dark              | 2023-01-18 |
# | 5             | Arthur    | GDC (F)        | 721.1       | Light         | Sunset->Dark      | 2023-01-18 |
# | 6             | Pranav    | WCP (B)        | 1358.2      | Medium        | Sunny             | 2023-01-26 |
# | 7             | Pranav    | Guad (F)       | 1619        | Medium        | Sunny             | 2023-01-26 |
# | 8             | Chaitanya | UNB            | 1020.8      | Light         | Cloudy            | 2023-01-27 |
# | 9             | Christina | WCP(B)         | 1228.9      | Medium        | Dark              | 2023-01-27 |
# | 10            | Raymond   | WCP (F)        | 1331.4      | Medium        | Cloudy            | 2023-01-30 |
# | 11            | Christina | WCP (B)        | 1648.9      | Light         | Sunny             | 2023-02-03 |
# | 12            | Pranav    | Guad (B)       | 1968.7      | Light         | Cloudy            | 2023-02-03 |
# | 13            | Christina | WCP (F)        | 1539.4      | Heavy         | Cloudy/Rainy      | 2023-02-06 |
# | 14            | Chaitanya | UNB            | 1198.6      | Light         | Cloudy            | 2023-02-06 |
# | 15            | Chaitanya | UNB            | 1133.8      | Medium        | Dark/Rainy        | 2023-02-07 |
# | 16            | Pranav    | Guad (F)       | 1706.2      | Heavy         | Cloudy/Rainy      | 2023-02-08 |
# | 17            | Lochana   | Guad (B)       | 1951.9      | Medium        | Sunny             | 2023-02-08 |
# | 18            | Pranav    | GDC (B)        | 876.3       | Medium        | Sunny             | 2023-02-09 |
# | 19            | Lochana   | GDC (F)        | 1443.9      | Medium        | Sunny             | 2023-02-09 |
# | 20            | Raymond   | WCP (F)        | 1402.7      | Heavy         | Sunny             | 2023-02-09 |
# | 21            | Arthur    | Guad (B)       | 2152.9      | Light         | Cloudy/Clear      | 2023-02-10 |
# | 22            | Chaitanya | WCP (F)        | 1551.5      | Medium        | Sunny             | 2023-02-10 |
# GDC 7, Guad 6, WCP 7, UNB 3

is_continous = True
label_path = './data/CODA/3d_bbox/os1'
label_dirs = os.listdir(label_path)

train_val_split = 0.6
train_scenes = []
val_scenes = []
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
        if frames[i] - frames[i-1] == 1 or not is_continous:
            sequence.append(frames[i])
        else:
            sequences.append(sequence)
            sequence = [frames[i]]
    sequences.append(sequence)

    random.shuffle(sequences)
    count = 0
    for i, sequence in enumerate(sequences):
        if count / len(frames) < train_val_split and i != len(sequences) - 1:
            train_scenes.append({
                'scene': label_dir,
                'frames': sequence,
                'frame_len': len(sequence)
            })
        else:
            val_scenes.append({
                'scene': label_dir,
                'frames': sequence,
                'frame_len': len(sequence)
            })
        count += len(sequence)

train_info = {
    'scenes': train_scenes,
    'scene_len': len(train_scenes),
    'frame_len': sum([scene['frame_len'] for scene in train_scenes])
}
val_info = {
    'scenes': val_scenes,
    'scene_len': len(val_scenes),
    'frame_len': sum([scene['frame_len'] for scene in val_scenes])
}

save_file = './data/CODA/split.json'
with open(save_file, 'w') as f:
    json.dump({
        'train': train_info,
        'val': val_info
    }, f, indent=4)
        
# random.shuffle(scenes)
# cusum = [0]
# for scene in scenes:
#     cusum.append(cusum[-1] + scene['frame_len'])
# percents = [item / cusum[-1] for item in cusum]

# train_val_split = 0.5
# train_val_split_idx = 0
# for i in range(len(percents)):
#     if percents[i] > train_val_split:
#         train_val_split_idx = i
#         break
# train_scenes = scenes[:train_val_split_idx]
# val_scenes = scenes[train_val_split_idx:]