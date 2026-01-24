Python version: 3.12.12

To create a virtual environment using python:

```bash
# macOS/Linux:
python3.12 -m venv ml-exercise-group19

# Windows:
py -3.12 -m venv ml-exercise-group19
```

To activate the environment:

```bash
# macOS/Linux:
source ml-exercise-group19/bin/activate

# Windows (Command prompt):
ml-exercise-group19\Scripts\activate
```

To install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
To verify installation:

```bash
python test_dependencies.py
```
Or run one-by-one if it doesn't work:

```bash
pip install _
```
Additionally, if the system has a cuda device (https://developer.nvidia.com/cuda/gpus), you can install the specific pytorch-cuda version (https://pytorch.org/get-started/locally/) and the GPU will be used by default by all deep learning training scripts. 

To deactivate the environment:

```bash
deactivate
```

Otherwise, use preferred environment / venv manager and install "requirements.txt".


(2) Deep Learning

First navigate to src/training. "train.py" uses "config.yml" by default, but using the "--config" argument allows you to select any preset you want. We create presets for every model - dataset - augmentation combination and they can be run with the following commands:

```bash
python train.py --config ../../config/sqn_c10_aug.yml
python train.py --config ../../config/sqn_c10_noaug.yml
python train.py --config ../../config/sqn_fmnst_aug.yml
python train.py --config ../../config/sqn_fmnst_noaug.yml
python train.py --config ../../config/rnt_c10_aug.yml
python train.py --config ../../config/rnt_c10_noaug.yml
python train.py --config ../../config/rnt_fmnst_aug.yml
python train.py --config ../../config/rnt_fmnst_noaug.yml
```
The models will be saved in the root /checkpoints folder.

Note that the ResNet transfer learning models are quite computationally expensive for CPU training and might need less epochs / downsized loaded model to train in a reasonable time frame.

To evaluate the checkpoints, navigate to src/utils. The "eval.py" script takes "--config" and "--checkpoint" as arguments. The checkpoint file has to match the config file with which the model was trained. You can evaluate the model presets with the following commands:

```bash
python eval.py --config ../../config/sqn_c10_aug.yml --checkpoint ../../squeezenet_cifar10_aug
python eval.py --config ../../config/sqn_c10_noaug.yml --checkpoint ../../squeezenet_cifar10_noaug
python eval.py --config ../../config/sqn_fmnst_aug.yml --checkpoint ../../squeezenet_fmnist_aug
python eval.py --config ../../config/sqn_fmnst_noaug.yml --checkpoint ../../squeezenet_fmnist_noaug
python eval.py --config ../../config/rnt_c10_aug.yml --checkpoint ../../resnet_cifar10_aug
python eval.py --config ../../config/rnt_c10_noaug.yml --checkpoint ../../resnet_cifar10_noaug
python eval.py --config ../../config/rnt_fmnst_aug.yml --checkpoint ../../resnet_fmnist_aug
python eval.py --config ../../config/rnt_fmnst_noaug.yml --checkpoint ../../resnet_fmnist_noaug
``` 
The results will be saved in the root /evals folder.

