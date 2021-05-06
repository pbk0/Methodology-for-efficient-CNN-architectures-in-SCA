## Modifications

This branch `attack_original` is same as `master` branch.

The only difference is that we have commented training code 
in each experiment and just loaded the already provided model 
files and again ran all the experiments.

## Installation instructions

Note that the original code is old and will need a python 3.7 installation.
Plus make sure that you have correct versions of some libraries (mentioned below)
The code is not compatible with newer version so consider downgrading some libraries.

```bat
pip install tensorflow==1.13.1
pip install keras==2.2.4
pip install h5py==2.10.0
pip install numpy==1.16.4
pip install scipy==1.5.4
```

Other libraries that will be needed

```bat
pip install -U scikit-learn
pip install matplotlib
```

## Comparing results

### AES_HD

Branch: `master`           |  Branch `attack_original`
:-------------------------:|:-------------------------:
![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/master/AES_HD/fig/rankAES_HD_1500trs_100att.svg)  |  ![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/attack_original/AES_HD/fig/rankAES_HD_1500trs_100att.svg)



### AES_RD

Branch: `master`           |  Branch `attack_original`
:-------------------------:|:-------------------------:
![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/master/AES_RD/fig/rankAES_RD_15trs_100att.svg)  |  ![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/attack_original/AES_RD/fig/rankAES_RD_15trs_100att.svg)



### ASCAD desync=0

Branch: `master`           |  Branch `attack_original`
:-------------------------:|:-------------------------:
![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/master/ASCAD/N0%3D0/fig/rankASCAD_desync0_300trs_100att.svg)  |  ![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/attack_original/ASCAD/N0%3D0/fig/rankASCAD_desync0_300trs_100att.svg)



### ASCAD desync=50

Branch: `master`           |  Branch `attack_original`
:-------------------------:|:-------------------------:
![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/master/ASCAD/N0%3D50/fig/rankASCAD_desync50_400trs_100att.svg)  |  ![](https://github.com/SpikingNeuron/Methodology-for-efficient-CNN-architectures-in-SCA/blob/attack_original/ASCAD/N0%3D50/fig/rankASCAD_desync50_400trs_100att.svg)

