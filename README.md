# CIFAR10

![Version](https://img.shields.io/badge/version-v0.4.1-blue?style=flat-square)

Read the [ai-project docs](https://gradiant.github.io/ai-project-template/).

## Fist steps
### Create conda environment

Create a conda environment from [conda.yml](conda.yml) and activate it:

```bash
conda env create -f environment/conda.yaml
. activate [environment-name]
```

### Install pre-commit hooks

```bash
. activate ai-project-cifar10

pre-commit install
```

### DVC

First you must checkout and reproduce DVC state. It can be done by runing the following commands:

```bash
. activate ai-project-cifar10

dvc checkout

dvc repro
```
