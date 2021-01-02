# Pedestrian Crossing Action Prediction Benchmark

Benchmark for evaluating pedestrian action prediction algorithms that inlcude code for training, testing and evaluating baseline and state-of-the-art models for pedestrian action prediction on PIE and JAAD datasets.

**Paper: [I. Kotseruba, A. Rasouli, J.K. Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021.](https://openaccess.thecvf.com/content/WACV2021/papers/Kotseruba_Benchmark_for_Evaluating_Pedestrian_Action_Prediction_WACV_2021_paper.pdf)** (see [citation](#citation) information below)


# Installation instructions
1. Download and extract PIE and JAAD datasets.
	
	Follow the instructions provided in [https://github.com/aras62/PIE](https://github.com/aras62/PIE) and [https://github.com/ykotseruba/JAAD](https://github.com/ykotseruba/JAAD).

2. Download Python data interface.

	Copy `pie_data.py` and `jaad_data.py` from the corresponding repositories into `Ped_Cross_Benchmark` directory.

3. Install docker (see instructions for [Ubuntu 16.04](https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/) and [Ubuntu 20.04](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04)).

4. Change permissions for scripts in `docker` folder:
	```
	chmod +x docker/*.sh
	```

5. Build docker image

	```
	docker/build_docker.sh
	```

	Optionally, you may set custom image name and/or tag using this command (e.g. to use two GPUs in parallel):
	```
	docker/build_docker.sh -im <image_name> -t <tag>
	```

6. Get optical flow data

	- Use [Flownet2](https://github.com/lmb-freiburg/flownet2) to generate optical flow for I3D.

# Running instructions using Docker

## Run container in interactive mode:

Set paths for PIE and JAAD datasets in `docker/run_docker.sh` (see comments in the script).

Then run:

```
./docker/run_docker.sh
```

### Train and test a model with default parameters

Use `train_test.py` script with `model_name` and `dataset` arguments (see valid options below):
```
python train_test.py -m <model_name> -d <dataset>
```

For example, to train SFRNN model with default parameters on the JAAD dataset run:  

```
python train_test.py -c config_files/SFRNN.yaml -d jaad_all
```

The script will automatially save the trained model weights, configuration file and evaluation results in the `model/<model_name>/<current_date>/` folder.


Command line argument options:

`model_name`: 

- ATGC
- C3D
- ConvLSTM_resnet50
- ConvLSTM_vgg16
- HierarchicalRNN
- I3D (RGB only)
- I3D_flow (optical flow)
- MultiRNN
- PCPA
- SFRNN
- SingleRNN_gru
- SingleRNN_lstm
- StackedRNN
- Static_resnet50
- Static_vgg16
- Two_Stream

`dataset`: 

- pie
- jaad\_beh (subset of JAAD containing pedestrian samples with behavioral annotations)
- jaad\_all (the entire JAAD dataset)

### Train and test a model with custom parameters

Use `train_test.py` script with path to the YAML config file:
```
python train_test.py -c <path_to_config_file> -d <dataset>
```

Custom configuration file may overwrite default parameters specified in `config_files/configs_default.yaml`. See comments in the default config file and `action_predict.py` for parameter descriptions.

### Test saved model

To re-run test on the saved model use:

```
python test_model.py <saved_files_path>
```
<a name="citation"></a>
## Citation

If you use the results, analysis or code for the models presented in the paper, please cite:

```
@inproceedings{kotseruba2021benchmark,
	title={{Benchmark for Evaluating Pedestrian Action Prediction}},
	author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
	booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
	pages={1258--1268},
	year={2021}
}
```

If you use model implementations, please cite the corresponding papers

- ATGC [1] 
- C3D [2]
- ConvLSTM [3]
- HierarchicalRNN [4]
- I3D [5]
- MultiRNN [6]
- PCPA [7]
- SFRNN [8] 
- SingleRNN [9]
- StackedRNN [10]
- Two_Stream [11]

[1] Amir Rasouli, Iuliia Kotseruba, and John K Tsotsos. Are they going to cross?  A benchmark dataset and baseline for pedestrian crosswalk behavior.  ICCVW, 2017.

[2] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani,and Manohar Paluri. Learning spatiotemporal features with 3D convolutional networks. ICCV, 2015.

[3] Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung,Wai-Kin Wong, and Wang-chun Woo. Convolutional LSTM network:  A machine learning approach for precipitation nowcasting. NeurIPS, 2015.

[4] Yong Du, Wei Wang, and Liang Wang. Hierarchical recurrent neural network for skeleton based action recognition. CVPR, 2015

[5] Joao Carreira and Andrew Zisserman.  Quo vadis, action recognition?  A new model and the kinetics dataset.  CVPR, 2017.

[6] Apratim Bhattacharyya, Mario Fritz, and Bernt Schiele. Long-term on-board prediction of people in traffic scenes under uncertainty. CVPR, 2018.

[7] Iuliia Kotseruba, Amir Rasouli, and John K Tsotsos, Benchmark for evaluating pedestrian action prediction. WACV, 2021.

[8] Amir Rasouli, Iuliia Kotseruba, and John K Tsotsos. Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs. BMVC, 2019

[9] Iuliia Kotseruba, Amir Rasouli, and John K Tsotsos.  Do They Want to Cross? Understanding Pedestrian Intention for Behavior Prediction. In IEEE Intelligent Vehicles Symposium (IV), 2020.

[10] Joe Yue-Hei Ng, Matthew Hausknecht, Sudheendra Vi-jayanarasimhan, Oriol Vinyals, Rajat Monga, and GeorgeToderici. Beyond short snippets: Deep networks for video classification. CVPR, 2015.

[11] Karen Simonyan and Andrew Zisserman. Two-stream convolutional networks for action recognition in videos. NeurIPS, 2014.