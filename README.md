# Hand Detection and Orientation Estimation
This project utilizes a modified MobileNet in company with the [SSD](https://github.com/weiliu89/caffe/tree/ssd) framework to achieve a robust and fast detection of hand location and orientation. 
Our implementation is based on [the PyTorch version of SSD](https://github.com/amdegroot/ssd.pytorch) and [MobileNet](https://github.com/ruotianluo/pytorch-mobilenet-from-tf).

<img src="https://github.com/yangli18/hand_detection/blob/master/data/results/demo/010174_hand.svg" height=216><img src="https://github.com/yangli18/hand_detection/blob/master/data/results/demo/010061_hand.svg" height=216><img src="https://github.com/yangli18/hand_detection/blob/master/data/results/demo/010210_hand.svg" height=216>

## Contents
1. [Preparation](#preparation)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Citation](#citation)


## Preparation
1. Due to some compatibility issues, we recommend to install PyTorch 0.3.0 and Python 3.6.8, which our project currently supports. 

2. Get the code. 
    ```Shell
    git clone https://github.com/yangli18/hand_detection.git
    ```
3. Download [the Oxford hand dataset](http://www.robots.ox.ac.uk/~vgg/data/hands/) and create the LMDB file for the training data.
    ```Shell
    sh data/scripts/Oxford_hand_dataset.sh
    ```
4. Compile the NMS code (from [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn/tree/0.3)).
    ```Shell
    sh layers/src/make.sh
    ```


## Training

Train the detection model on the Oxford hand dataset. 
```Shell
python train.py 2>&1 | tee log/train.log
```
* Trained with data augmentation v2, our detection model reaches over **86%** average precision (AP) on the Oxford hand dataset. We provide the trained model in the `weights` dir.
* For the convenience of training, a pre-trained MobileNet is put in the `weights` dir. 
You can also download it from [here](https://github.com/ruotianluo/pytorch-mobilenet-from-tf).
 

## Evaluation

1. Evaluate the trained detection model.
    ```Shell
    python eval.py --trained_model weights/ssd_new_mobilenet_FFA.pth --version ssd_new_mobilenet_FFA
    ```
    * ***Note***: For a fair comparison, the evaluation code of the Oxford hand dataset should be used to get the exact mAP (mean Average Precision) of hand detection. 
    The above command should return a similar result, but not exactly the same.
    
2. Evaluate the average detection time.
    ```Shell
    python eval_speed.py --trained_model weights/ssd_new_mobilenet_FFA.pth --version ssd_new_mobilenet_FFA
    ```   


## Citation
If you find our code useful, please cite our paper. 
```
@inproceedings{yang2018light,
  title={A Light CNN based Method for Hand Detection and Orientation Estimation},
  author={Yang, Li and Qi, Zhi and Liu, Zeheng and Zhou, Shanshan and Zhang, Yang and Liu, Hao and Wu, Jianhui and Shi, Longxing},
  booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
  pages={2050--2055},
  year={2018},
  organization={IEEE}
}

@article{yang2019embedded,
  title={An embedded implementation of CNN-based hand detection and orientation estimation algorithm},
  author={Yang, Li and Qi, Zhi and Liu, Zeheng and Liu, Hao and Ling, Ming and Shi, Longxing and Liu, Xinning},
  journal={Machine Vision and Applications},
  volume={30},
  number={6},
  pages={1071--1082},
  year={2019},
  publisher={Springer}
}
```
