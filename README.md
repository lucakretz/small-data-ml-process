# small-data-ml-process
This is the implementation of the ML-Process developed during my master thesis.


## Tasks

### Data Transformation

```sh
python main.py -c configs/transformation_config.json run_transformation
```
### Data Set Creation

```sh
python main.py -c configs/creation_config.json run_creation
```

### Train Fraud Analysis

```sh
python main.py -c configs/training_config_fraud.json run_training
```

### Train Image Recognition

```sh
python main.py -c configs/training_config_recognition.json run_training
```

### Train Anomaly Detection

```sh
python main.py -c configs/training_config_detection.json run_training
```
