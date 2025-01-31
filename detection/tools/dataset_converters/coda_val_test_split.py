import random
import mmengine

info_path = './data/CODA/_coda_infos_val.pkl'
info = mmengine.load(info_path)

val_info = {'metainfo': info['metainfo'], 'data_list': []}
test_info = {'metainfo': info['metainfo'], 'data_list': []}
test_info['metainfo']['split'] = 'test'

split = 0.5
frame_num = len(info['data_list'])
val_frame_num = int(frame_num * split)
test_frame_num = frame_num - val_frame_num
frame_indices = list(range(frame_num))
random.shuffle(frame_indices)
val_frame_indices = frame_indices[:val_frame_num]
test_frame_indices = frame_indices[val_frame_num:]

for index in val_frame_indices:
    val_info['data_list'].append(info['data_list'][index])
    
for index in test_frame_indices:
    test_info['data_list'].append(info['data_list'][index])
    
mmengine.dump(val_info, './data/CODA/coda_infos_val.pkl')
mmengine.dump(test_info, './data/CODA/coda_infos_test.pkl')