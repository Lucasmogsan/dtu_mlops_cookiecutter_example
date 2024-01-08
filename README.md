# dtu_mlops_cookiecutter_example

simple image classification from dtu course

Things implemented:
1. Cookiecutter
2. dvc
3. docker
4. pytest

## Running code

Clone repo and pull data
```bash
git clone <my_repository>
cd <my_repository>
dvc pull
```


In package root from terminal:

Process data:
```bash
python dtu_mlops_cookiecutter_example/data/make_dataset.py
```

Train:
```bash
python dtu_mlops_cookiecutter_example/train_model.py train
```

Evaluate:
```bash
python dtu_mlops_cookiecutter_example/train_model.py evaluate model.pt
```

Predict:
```bash
python dtu_mlops_cookiecutter_example/predict_model.py predict model.pt data/processed/test_images.pt
python dtu_mlops_cookiecutter_example/predict_model.py predict model.pt data/np_image_test.npy
```

# Docker

See [docker README](dockerfiles/README.md)

# Pytest (and using coverage)

Put ```test_XX.py``` files in ```tests``` folder.
To run tests:
```bash
pytest tests/<specify-test-file-if-wanted>
```

Measure the **code coverage** (i.e. the percentage of your codebase that actually gets run when all your tests are executed).


```bash
coverage run -m pytest tests/
```

```bash
coverage report -m
coverage report -m --omit "tests/*"
```


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── dtu_mlops_cookiecutter_example  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
