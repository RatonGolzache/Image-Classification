Python version: 3.12.12

To create a virtual environment using python:

```bash
**macOS/Linux:**
python3.12 -m venv ml-exercise-group19

**Windows:**
py -3.12 -m venv ml-exercise-group19
```

To activate the environment:

```bash
**macOS/Linux:**
source ml-exercise-group19/bin/activate

**Windows (Command prompt):**
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

To deactivate the environment:

```bash
deactivate
```

Otherwise, use preferred environment / venv manager and install "requirements.txt".

