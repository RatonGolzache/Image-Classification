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
python train.py --config sqn_c10_aug.yml
python train.py --config sqn_c10_noaug.yml
python train.py --config sqn_fmnst_aug.yml
python train.py --config sqn_fmnst_noaug.yml
python train.py --config rnt_c10_aug.yml
python train.py --config rnt_c10_noaug.yml
python train.py --config rnt_fmnst_aug.yml
python train.py --config rnt_fmnst_noaug.yml
```

