# fnc-1
Submission for the [Fake News Challenge](http://www.fakenewschallenge.org).
The pipeline consists of text preprocessing, feature extraction, test-split generation, minority-class oversampling, followed by training and prediction using [LightGBM](https://github.com/Microsoft/LightGBM).

## Installation
This implementation was built and tested on Python 3.5. Dependencies can be installed via the following commands:
```
pip install -r requirements.txt
python -m spacy download en
```

LightGBM needs to be installed from source, by following the instructions [here](https://github.com/Microsoft/LightGBM/tree/master/python-package).

In addition, the FNC dataset is included as a submodule, and should be downloaded by running the following commands:
```
git submodule init
git submodule update
```

## Usage
```
python fnc-1.py
```
During the first run, models and features will be generated for the first time, which may take several hours. Subsequent runs will use cached data, stored in the `caches` directory.