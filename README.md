# PoseNet_Planar
In this project, we investigate efficient small-sized convolutional neural networks (CNNs) for ego-motion estimation of micro air vehicles (MAVs) equipped with a downward-facing camera and inertial measurement unit (IMU), especially focusing on the robustness of estimation in fast maneuvers. 

# Pulication:
CNN-based Ego-Motion Estimation for Fast MAV Maneuvers (accepted by ICRA 2021)

Conference Proceeding: to be published

ArXiv Pre-print (with Appendix): https://arxiv.org/abs/2101.01841

Videos: https://youtu.be/BMdh6dmLgrM https://youtu.be/Uz9pNpn94jU

# Prerequisities
Python3 and PyTorch v1.1.0 (tested with v1.1.0 and v1.4.0)

MATLAB (tested with v2019b and v2020a)

# Usage
## 1.Dataset Generation 
(MATLAB)
image_generation_matlab/main.m
## 2.Network Training
python3 main.py
## 3.Pre-trained Modles
python3 main.py --use-pretrained
## 4.Running on Public Flight Dataset
python3 main.py --flight --show-img

# Authors and Maintainers:
Yingfu Xu (y.xu-6@tudelft.nl) and Prof. Dr. Guido de Croon from Micro Air Vehicle Laboratory (MAVLab), TU Delft, The Netherlands.

# Acknowledgements

We greatly thanks Ir. Nilay Y. Sheth (https://github.com/nilay994) for his supports in developing the quadrotor MAV with GPU and collecting the flight datasets. We also appreciate the author of https://github.com/ClementPinard/FlowNetPytorch upon the code structure of which we developed our code for network training. The Madgwick AHRS code in this repo is taken from https://github.com/morgil/madgwick_py, whose author is appreciated as well.
