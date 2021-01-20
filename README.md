# Simple Multi-Resolution Representation Learning for Human Pose Estimation

## Environment
The code is developed using Python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. We recommend to use Anaconda. We already developed using Anaconda2. You can follow the steps below to set up the Anaconda environment.
1. Install Anaconda2:
   Go to [https://www.anaconda.com/distribution/#download-section](https://www.anaconda.com/distribution/#download-section), copy ```${LINK}``` for appropriate version.
   ```
   curl -O ${LINK}
   bash ${DOWNLOADED_FILE}
   source ~/.bashrc
   ```
2. Create new environment:
   ```
   conda create --name SimMRPose python=3.6 -y
   ```
3. Activate the environment:
   ```
   conda activate SimMRPose
   ```

## Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
2. Disable cudnn for batch_norm:
   ```
   # for pytorch v0.4.0
   sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   # for pytorch v0.4.1
   sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
   ```
   **Note**: ```${PYTORCH}``` indicates the path where pytorch is installed.
3. Clone this repo, and the cloned directory is called as ```${MR_POSE_ROOT}```.
4. Install dependencies:
   ```
   easydict
   opencv-python==3.4.1.15
   Cython
   scipy
   pandas
   pyyaml
   json_tricks
   scikit-image
   tensorboardx   
   torchvision
   ```
   **Note**: You can follow the steps below to install dependencies in the Anaconda2 environment:
   ```
   conda install -c conda-forge easydict
   python3.6 -m pip install opencv-python==3.4.1.15
   conda install Cython
   conda install scipy
   conda install pandas
   conda install pyyaml
   conda install -c conda-forge json_tricks
   conda install scikit-image
   conda install -c conda-forge tensorboardx   
   ```
5. Make libs:
   ```
   cd ${MR_POSE_ROOT}/lib
   make
   ```
6. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   git clone https://github.com/cocodataset/cocoapi.git
   cd ${COCOAPI}/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   **Note**: ```${COCOAPI}``` indicates the path where the COCOAPI is cloned.

7. In the ```${MR_POSE_ROOT}``` directory, create folders: ```data``` (dataset directory), ```models``` (pre-trained models directory), ```log``` (tensorboard log directory), and ```output``` (training model output directory):

   ```
   mkdir data
   mkdir models
   mkdir log
   mkdir output
   ```

   Your directory tree should look like this:

   ```
   ${MR_POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── scripts
   ├── README.md
   ```

## Data preparation
**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. 2017 Test is needed for the validation on COCO test-dev. The person detection results of COCO val2017 and test-dev2017 to reproduce multi-person pose estimation results are available at [GoogleDrive](https://drive.google.com/drive/folders/1awA6pSH9VXTNDxpukaL5Fes8hnkgK4Dp).
Download and extract them under ```{MR_POSE_ROOT}/data```, and make them look like this:
```
${MR_POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- image_info_test2017.json
        |   |-- image_info_test-dev2017.json
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        |   `-- COCO_val2017_detections_AP_H_56_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ...
            |-- val2017
            |   |-- 000000000139.jpg
            |   |-- 000000000285.jpg
            |   |-- 000000000632.jpg
            |   |-- ...
            `-- test2017
                |-- 000000000016.jpg
                |-- 000000000019.jpg
                |-- 000000000057.jpg
                |-- ...
```

**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. The json format is available at [GoogleDrive](https://drive.google.com/drive/folders/1EAUiFtXeDh7koV-nDP6i0HGpzETB-TT6).
Download and extract them under ```{MR_POSE_ROOT}/data```, and make them look like this:
```
${MR_POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

## Pre-trained models
1. Download pytorch imagenet pre-trained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo) or [GoogleDrive](https://drive.google.com/drive/folders/1OsJE6rpXfvqtFcr6txyJxlMR8Tuj6fiT), and download caffe-style pre-trained models from [GoogleDrive](https://drive.google.com/drive/folders/1yJMSFOnmzwhA4YYQS71Uy7X1Kl_xq9fN?usp=sharing). 
2. Download ```pose_resnet_50_256x192.pth.tar```, ```pose_resnet_101_256x192.pth.tar```, and ```pose_resnet_152_256x192.pth.tar``` model from [OneDrive](https://onedrive.live.com/?authkey=%21AFkTgCsr3CT9%2D%5FA&id=56B9F9C97F261712%2110704&cid=56B9F9C97F261712) or [GoogleDrive](https://drive.google.com/drive/folders/1fGLeCgTbaO50wylfV_j1OFTEx8DDpaqh) to reproduce the results of [SimpleBaseline](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf) on COCO test-dev2017 dataset.
3. Download ```pose_resnet_50_256x192.pth.tar```, ```pose_resnet_101_256x192.pth.tar```,  and ```pose_resnet_152_256x192.pth.tar``` model from [OneDrive](https://onedrive.live.com/?authkey=%21AFkTgCsr3CT9%2D%5FA&id=56B9F9C97F261712%2110709&cid=56B9F9C97F261712) or [GoogleDrive](https://drive.google.com/drive/folders/1g_6Hv33FG6rYRVLXx1SZaaHj871THrRW) to reproduce the results of [SimpleBaseline](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf) on MPII test set.
4. Download our multi-resolution representation learning models trained on COCO and MPII dataset from [GoogleDrive](https://drive.google.com/drive/folders/1q0c4w3JYrgwI4Vm63OHmJa1EmfXNUBJy?usp=sharing)

Download them under ```${MR_POSE_ROOT}/models/pytorch```, and make them look like this:

   ```
   ${MR_POSE_ROOT}
    `-- models
        `-- pytorch
            |-- imagenet
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet50-caffe.pth.tar
            |   |-- resnet101-5d3b4d8f.pth
            |   |-- resnet101-caffe.pth.tar
            |   |-- resnet152-b121ed2d.pth
            |   `-- resnet152-caffe.pth.tar
            |-- mr_pose_coco
            |   |-- mrfea1_pose_resnet_50_256x192.pth.tar
            |   |-- mrfea2_pose_resnet_50_256x192.pth.tar
            |   |-- mrheat1_pose_resnet_50_256x192.pth.tar
            |   |-- mrheat2_pose_resnet_50_256x192.pth.tar
            |   |-- mrfea2_pose_resnet_101_256x192.pth.tar
            |   `-- mrfea2_pose_resnet_152_256x192.pth.tar
            |-- mr_pose_mpii
            |   |-- mrfea1_pose_resnet_50_256x192.pth.tar
            |   |-- mrfea1_pose_resnet_101_256x192.pth.tar
            |   |-- mrfea1_pose_resnet_152_256x192.pth.tar
            |   |-- mrfea2_pose_resnet_50_256x192.pth.tar
            |   |-- mrfea2_pose_resnet_101_256x192.pth.tar
            |   |-- mrfea2_pose_resnet_152_256x192.pth.tar
            |   |-- mrheat1_pose_resnet_50_256x192.pth.tar
            |   |-- mrheat1_pose_resnet_101_256x192.pth.tar
            |   |-- mrheat1_pose_resnet_152_256x192.pth.tar
            |   |-- mrheat2_pose_resnet_50_256x192.pth.tar
            |   |-- mrheat2_pose_resnet_101_256x192.pth.tar
            |   `-- mrheat2_pose_resnet_152_256x192.pth.tar
            |-- pose_coco
            |   |-- pose_resnet_50_256x192.pth.tar
            |   |-- pose_resnet_101_256x192.pth.tar
            |   `-- pose_resnet_152_256x192.pth.tar
            `-- pose_mpii
                |-- pose_resnet_50_256x192.pth.tar
                |-- pose_resnet_101_256x192.pth.tar
                `-- pose_resnet_152_256x192.pth.tar
   ```
   
## Training and Validating
**Note**: To run training or validating with multi-resolution heatmap learning networks (MRHeatNet1 and MRHeatNet2) and multi-resolution feature map leanrning networks (MRFeaNet1 and MRFeaNet2), set the field ```MODEL.NAME``` in the config file to ```mrheat1_pose_resnet```, ```mrheat2_pose_resnet```, ```mrfea1_pose_resnet```, and ```mrfea2_pose_resnet``` respectively. \
For example, set ```MODEL.NAME``` of the config file ```experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml``` as follows:
   ```
   MODEL:
      NAME: 'mrheat1_pose_resnet'
   ```

### Validating on COCO val2017 using trained models and detection results
Set ```DATASET.TEST_SET``` in ```experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml``` to ```val2017```.
```
bash scripts/coco2017/valid.sh experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml models/pytorch/mr_pose_coco/mrheat1_pose_resnet_50_256x192.pth.tar data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json
```

### Validating on COCO test-dev2017 using trained models and detection results
Set ```DATASET.TEST_SET``` in ```experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml``` to ```test-dev2017```.
* With our models:
```
bash scripts/coco2017/valid.sh experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml models/pytorch/mr_pose_coco/mrheat1_pose_resnet_50_256x192.pth.tar data/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json
```
* Reproduce results of [SimpleBaseline](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf):
Set ```MODEL.NAME``` in ```experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml``` to ```pose_resnet```.\
Use ```valid_baseline.sh``` instead of ```valid.sh```.
```
bash scripts/coco2017/valid_baseline.sh experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar data/coco/person_detection_results/COCO_test-dev2017_detections_AP_H_609_person.json
```

### Training on COCO train2017
Set ```DATASET.TEST_SET``` in ```experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml``` to ```val2017```.\
Set ```MODEL.NAME``` in ```experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml``` to ```mrheat1_pose_resnet```, ```mrheat2_pose_resnet```, ```mrfea1_pose_resnet```, and ```mrfea2_pose_resnet``` respectively.
```
bash scripts/coco2017/train.sh experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
```

### Validating on MPII using trained models
* With our models: 
Set ```MODEL.NAME``` in ```experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml``` to ```mrheat1_pose_resnet```, ```mrheat2_pose_resnet```, ```mrfea1_pose_resnet```, and ```mrfea2_pose_resnet``` respectively.
```
bash scripts/mpii/valid.sh experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml models/pytorch/mr_pose_mpii/mrheat1_pose_resnet_50_256x192.pth.tar
```
* Reproduce results of [SimpleBaseline](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf):
Set ```MODEL.NAME``` in ```experiments/mpii/resnet50/256x192_d256x3_adam_lr1e-3.yaml``` to ```pose_resnet```.\
Use ```valid_baseline.sh``` instead of ```valid.sh```.
```
bash scripts/mpii/valid_baseline.sh experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml models/pytorch/pose_mpii/pose_resnet_50_256x192.pth.tar
```

### Training on MPII
Set ```MODEL.NAME``` in ```experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml``` to ```mrheat1_pose_resnet```, ```mrheat2_pose_resnet```, ```mrfea1_pose_resnet```, and ```mrfea2_pose_resnet``` respectively.
```
bash scripts/mpii/train.sh experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```

## References
[1] B. Xiao, H. Wu, and Y. Wei, "Simple baselines for human pose estimation and tracking," in Proceedings of the European conference on computer vision (ECCV)
