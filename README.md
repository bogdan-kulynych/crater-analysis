
## Data

1. Export data from CVAT in the Imagenet format for tasks 5, 6, 7, 8
2. Put the zip archives into data/
3. In the data directory, run ./prepare.sh

## Setup

`pipenv install --dev`

## Training

`PYTHONPATH=. pipenv run python scripts/train.py`
