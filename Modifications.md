## Modifications

This branch `fixed_key_bias` is intended to do experiment with only `DPA-contest v4`
This is because it has multiple fixed key datasets.

We have uploaded all 16 datasets with attack and profile traces.
[Download link](https://drive.google.com/file/d/1Ol4fcvq2Nu0pRLn-yRu7vptQ-xTywgP7/view?usp=sharing).

Note that training will happen on datasets we provide as it is not possible to replicate the steps from 
original authors. We can verify things are still similar based on rank plots.

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

## Train on one dataset and attack on all 16 datasets

### Neural-Net was also trained on the `set_00`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_00.svg)
### Neural-Net was also trained on the `set_01`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_01.svg)
### Neural-Net was also trained on the `set_02`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_02.svg)
### Neural-Net was also trained on the `set_03`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_03.svg)
### Neural-Net was also trained on the `set_04`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_04.svg)
### Neural-Net was also trained on the `set_05`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_05.svg)
### Neural-Net was also trained on the `set_06`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_06.svg)
### Neural-Net was also trained on the `set_07`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_07.svg)
### Neural-Net was also trained on the `set_08`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_08.svg)
### Neural-Net was also trained on the `set_09`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_09.svg)
### Neural-Net was also trained on the `set_10`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_10.svg)
### Neural-Net was also trained on the `set_11`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_11.svg)
### Neural-Net was also trained on the `set_12`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_12.svg)
### Neural-Net was also trained on the `set_13`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_13.svg)
### Neural-Net was also trained on the `set_14`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_14.svg)
### Neural-Net was also trained on the `set_15`
![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_15.svg)


## Train on a mix of all 16 dataset's and attack on all 16 datasets

![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_all.svg)

## Train on a mix of first four dataset's and attack on all 16 datasets

![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_0123.svg)

## Train on a mix of set_00 and set_11 dataset's and attack on all 16 datasets

![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_0and11.svg)

## Train on a mix of three datasets that generalize well i.e. set_01, set_12 and set_13 dataset's and attack on all 16 datasets

![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_3good.svg)

## Train on a mix of three datasets that generalize bad i.e. set_10, set_11 and set_15 dataset's and attack on all 16 datasets

![](./DPA-contest%20v4/fig/rank_all_sets_with_100_attacks_for_model_DPA-contest_v4_trained_on_set_3bad.svg)