# An Efficient LiDAR-Camera Fusion Network for Multi-Class 3D Dynamic Object Detection and Trajectory Prediction [arXiv](https://arxiv.org/abs/2504.13647)

## 1. Get Started
### 1.1 Prerequisites
Ubuntu 20.04 \
CUDA 11.x or 12.x \
Python 3.10

### 1.2 Python Environment
**Step 1.**  Install PyTorch
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

**Step 2.**  Install MMEngine, MMCV, MMDetection and MMDetection3D using MIM
```shell
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmdet3d==1.4.0
```

**Step 3.**  Install spconv and flash-attn
```shell
pip install spconv-cu120==2.3.6
pip install flash-attn==2.6.3
```
You can visit https://github.com/traveller59/spconv to choose the suitable version of spconv for you.  
Installing flash-attn directly via pip command may meet error. You can visit https://github.com/Dao-AILab/flash-attention to download the corresponding wheel package, and then install flash-attn locally by pip command.

**Step 4.**  Install other packages and change numpy version
```shell
pip install filterpy
pip install numpy==1.26.4
```

**Step 5.**  Install Mamba
```shell
pip install causal-conv1d>=1.1.0
cd detection/my_projects/UniMT/unimt/mamba
pip install .
```

### 1.3 CODA Dataset
The UT Campus Object Dataset (CODa) is provided in [this paper](https://arxiv.org/pdf/2309.13549). You can download the dataset [here](https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa). It's recommended to choose split/CODa_full_split.zip to be consistent with us. Once you have downloaded and decompressed the dataset, link it to the directories "\${workdir}/detection/data/CODA", "\${workdir}/tracking/data/CODA" and "\${workdir}/trajectory_prediction/data/CODA".  

## 2. Detection
We provided all the code for CODA deteciton experiments in our paper. You can follow the instructions below to train the corresponding model or ues the model parameters we have given for test.  

### 2.1 Dataset Preparation
Download [coda_info](https://drive.google.com/drive/folders/1mDJrAE1o7uuFm7XMZn1EDI_BLQ6vDH6m?usp=sharing).    
* "coda_infos_train.pkl", "coda_infos_val.pkl" and "coda_infos_test.pkl" are used for train and test detection models.  
* "_coda_infos_val.pkl" contains data infos of "coda_infos_val.pkl" and "coda_infos_test.pkl", which is used for tracking and trajectory prediction.  
* Put these files in "\${workdir}/detection/data/CODA".  

### 2.2 Train
```shell
cd detection
./tools/dist_train.sh my_projects/${model}/configs/${config_file} ${gpu_num}
```
* You can replace \${model} to UniMT, CMT, BEVFusion or CenterPoint, then replace \${config_file} to corresponding file that ends with "coda.py".  
* If you want to train UniMT model, download pretrained backbone [convnextv2_nano.pth](https://drive.google.com/drive/folders/1PXP8glbf5VoRDix-hlE0TBfx-48icKI7?usp=sharing) at first and put it in folder "\${workdir}/detection/ckpts/pretrain".  
* If you want to train CMT model, download pretrained backbone [nuim_r50.pth](https://drive.google.com/drive/folders/1PXP8glbf5VoRDix-hlE0TBfx-48icKI7?usp=sharing) at first and put it in folder "\${workdir}/detection/ckpts/pretrain".  
* If you want to train BEVFusion model, download pretrained backbone [swin_tiny_patch4_window7_224.pth](https://drive.google.com/drive/folders/1PXP8glbf5VoRDix-hlE0TBfx-48icKI7?usp=sharing) at first and put it in folder "\${workdir}/detection/ckpts/pretrain".  
* We train these models on two Nvidia 3090 GPUs. You may need to adjust the learing rate in \${config_file} based on the number of GPUs you are using.  

### 2.3 Test
```shell
cd detection
./tools/dist_test.sh my_projects/${model}/configs/${config_file} ${checkpoint} ${gpu_num}
```
* The selection of \${model}, \${config_file} is the same as above.  
* You can download [checkpoints](https://drive.google.com/drive/folders/1PXP8glbf5VoRDix-hlE0TBfx-48icKI7?usp=sharing) and get the same results to our paper.  
* Due to the inherent randomness of the detection model, the results may have little difference from those in the paper.

### 2.4 Detection experiment results on CODa
|     Methods     | AP(Pedestrian) |   AP(Car)   | AP(Cyclist) |    mAP    |
| :-------------: | :------------: | :---------: | :---------: | :-------: |
|   CenterPoint   |     75.56 %    |   52.56 %   |   60.64 %   |  62.92 %  |
|    BEVFusion    |     79.55 %    |   54.18 %   |   63.74 %   |  65.82 %  |
|       CMT       |     80.87 %    |   62.41 %   |   66.38 %   |  69.89 %  |
|      UniMT      |     81.42 %    |   66.25 %   |   73.14 %   |  73.60 %  |

### 2.5 Detection experiment results on nuScenens
|  Methods  |  mATE  |  mASE  |  mAOE  |  mAVE  |  mAAE  |   mAP   |   NDS   |
| :-------: | :----: | :----: | :----: | :----: | :----: | :-----: | :-----: |
|   UniMT   |  23.9  |  23.4  |  27.4  |  23.3  |  12.7  |   72.7  |   75.3  |
* You can find our detailed test results at the official evaluation [address](https://evalai.s3.amazonaws.com/media/submission_files/submission_530484/67e5fa23-2b66-4547-a6ee-e285d73d750a.json).

## 3. Tracking
### 3.1 Dataset Preparation
You need generate detection results first. Prepare the coda_info and checkpoint of UniMT model, then run the command below.  
```shell
cd detection
python tools/infer_eval.py
```
* You will get detection results file "coda_unimt_detect_results.pkl". Then put "_coda_infos_val.pkl" and "coda_unimt_detect_results.pkl" in folder "\${workdir}/tracking/data/CODA".  

### 3.2 Test
```shell
cd tracking
python tools/infer_eval.py
```
* With the command above, you can obtain the results of the SimpleTrack method using DIoU and GPU.  
* If you want to test original SimpleTrack method, replace the "config_path" to './configs/coda_configs/giou.yaml' in file "\${workdir}/tracking/tools/infer_eval.py"  

## 4. Trajectory Prediction
### 4.1 Dataset Preparation
Download [split.json](https://drive.google.com/drive/folders/1mDJrAE1o7uuFm7XMZn1EDI_BLQ6vDH6m?usp=sharing), then put it in folder "\${workdir}/trajectory_prediction/data/CODA". Run the command below to generate trajectory data for train and val.  
```shell
cd trajectory_prediction
python tools/create_data_coda.py
```

### 4.2 Test on val dataset
The code already includes the parameters of the trained model named “coda_best.pth” in folder "\${workdir}/trajectory_prediction/checkpoints". You can directly test it.  
```shell
cd trajectory_prediction
python test.py
```

### 4.3 Train
```shell
cd trajectory_prediction
python train.py
```

#### The train and test of other methods in ours paper requires much additional code and data processing. We have not released the code at this time. If you are interested in this code, please feel free to contact me directly.  

### 4.4 Trajectory prediction experiment results
|     Methods     | ADE(Pedestrian)<sub>3/5/10 | ADE(Car)<sub>3/5/10 | ADE(Cyclist)<sub>3/5/10| FDE(Pedestrian)<sub>3/5/10 | FDE(Car)<sub>3/5/10 | FDE(Cyclist)<sub>3/5/10|
| :-------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|   Social-GAN    | 0.2975/0.2791/0.2586 | 1.1160/1.0673/1.0144 | 1.6706/1.5358/1.3866 | 0.5218/0.4821/0.4369 | 2.0744/1.9656/1.8502 | 3.2201/2.9324/2.6106 |
| Social-Implicit | 0.3532/0.3358/0.3164 | 0.8158/0.7998/0.7811 | 1.1291/1.1107/1.0889 | 0.6175/0.5816/0.5402 | 1.5213/1.4878/1.4473 | 2.1216/2.0855/2.0399 |
|      Ours       | 0.2667/0.2383/0.2083 | 0.3894/0.3519/0.3143 | 0.8490/0.7785/0.7174 | 0.4664/0.4058/0.3390 | 0.6977/0.6138/0.5229 | 1.7548/1.5888/1.4520 |

## 5. Complete Task Test and Visualization
```shell
python infer/test.py
```

#### For the deployment of this algorithm on ROS, please refer to my another [repository](https://github.com/TossherO/ros_packages).
