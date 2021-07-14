# small-data-ml-process
This is the implementation of the ML-Process developed during my master thesis.

## Setup
1. Create a conda invironment:
```sh
conda create -n pipeline_venv python=3.7.4
```
2. Activate environment:
```sh
conda activate pipeline_venv
```
3. Install required libraries:
```sh
pip install -r requirements.txt
```
## Project Structure
```
.
├── LICENSE
├── README.md
├── configs
│   ├── creation_config.json
│   ├── training_config_detection.json
│   ├── training_config_fraud.json
│   ├── training_config_recognition.json
│   └── transformation_config.json
├── data_processing
│   ├── __init__.py
│   ├── abstract_classes
│   │   ├── __init__.py
│   │   ├── abstract_creator.py
│   │   ├── abstract_extractor.py
│   │   └── abstract_reader.py
│   ├── data_annotator.py
│   ├── data_collector.py
│   ├── data_creator.py
│   ├── data_reader.py
│   ├── extractors
│   │   ├── __init__.py
│   │   └── path_extractor.py
│   ├── helpers
│   │   ├── __init__.py
│   │   ├── input_handler.py
│   │   └── type_handler.py
│   └── schemas.py
├── main.py
├── model_training
│   ├── annotation_handler.py
│   ├── data_module.py
│   ├── data_set.py
│   ├── model_module.py
│   ├── trainer_module.py
│   └── transformation.py
├── requirements.txt
└── utils
    ├── __init__.py
    ├── console_utils.py
    ├── converter_utils.py
    ├── io_utils.py
    └── parser.py
```

## Tasks
The end-to-end process encompasses the following steps:
- Data Transformation: Reading training data and converting the annotations to the required format.
- Data Set Creation: Splitting the transformed data to a train, validation and test set.
- Model Training: There are three exemplary pipelines (image recognition, fraud analysis and anomaly detection) available. This step automatically saves the best model based on validation metric and produces final metrics on test set.

### Data Transformation
#### Execution
```sh
python main.py -c configs/transformation_config.json run_transformation
```

#### Config
To run the Data Transformation, first the config file needs to be adjusted:
```json
{
    "data": {
        "in_path": "./path/to/image/data",
        "out_path": "./path/to/processed/data",
        "collector_mode": "tree",
        "type": "annotations",
        "transfer_files": false
    }
}
```
_in_path_: Path for the image data to transform.  
_out_path_: Folder to store the transformation results.  
_collector_mode_: Structure the input data is stored. Options: 
- `tree`: Read images from all subsequent folders of _in_path_.
- `folder`: Read images from given folder.
- `file`: Read on single file (_in_path_ needs to be an image file path).

_type_: Type of data to read (relevant for different uses of the __Reader__ module). Options:  
- `annotations`: Read image data and assign annotations or read annotation files.
- `data`: Read only images.

_transfer_files_: If `true`, images are transferred to folder given as _out_path_ (helpful for equally named images across different classes).

#### Process
- Execute the main.py using the _transformation_config.json_
- All data in given directory is listed -> Choose the file types to process (i.e.: _png_)
- Select a subset of data (if all data should be taken into account, type in 'no subset')
- Select an instance in the path to assign annotations on. If this is not possible and the labels should be assigned based on the file name, provide a label map in the following style: _string_0:0;string_1:1; ..._  
This leads to all images with the substring 'string_0' would get the label '0'. If there are images that match no substring, they get the `None` label.
- The assigned annotations are saved as _json_ files in the following style:
```json
{"0": {"type": "label", "class": "dog", "encoding": 0}}
```

#### Contributing
This process can easily be extended by adding a new __Data Extractor__ module [here](./data_processing/extractors).
1. Create a _xxx_extractor.py_ file
2. Set up class XXXExtractor by inheriting from [AbstractExtractor](./data_processing/abstract_classes/abstract_extractor.py) class.
3. Implement a _process_information_ and a _extract_ method. 
    - process_information: Prepare the files or get additional information for labeling
    - extract: Assign a label/annotation to each image path.
4. Add new extractor to `EXTRACTOR` variable in the [Reader](./data_processing/data_reader.py) script. This variable maps a given input data type to an extractor.
5. Optional: Extend the `MODES` variable in the [Reader](./data_processing/data_reader.py) script if a new annotation file format is added.

#### Current Implementation
- PathExtractor: Assignes labels to images based on folder or file names.


### Data Set Creation

#### Execution
```sh
python main.py -c configs/creation_config.json run_creation
```

#### Config
To run the Data Transformation, first the config file needs to be adjusted:
```json
{
    "data": {
        "in_path": "./path/to/images",
        "out_path": "./path/to/folder/with/split",
        "collector_mode": "tree",
        "type": "data",
        "annotation_folder": "./path/to/annotation/files",
        "data_split":{
            "train": 0.8,
            "val": 0.1,
            "test": 0.1
        }
    }
}
```
_in_path_: Path for the image data to split up for train, validation and testing.  
_out_path_: Folder to store the train, validation and test set.  
_collector_mode_: Structure the input data is stored. Options: 
- `tree`: Read images from all subsequent folders of _in_path_.
- `folder`: Read images from given folder.
- `file`: Read on single file (_in_path_ needs to be an image file path).

_type_: Type of data to read (relevant for different uses of the __Reader__ module). Options:  
- `annotations`: Read image data and assign annotations or read annotation files.
- `data`: Read only images.

_annotation_folder_: Folder with previously created annotation files.
_data_split_: Ratio of train, validation and test set. Has to add up to 1.

#### Process
- Execute the main.py using the _creation_config.json_
- All data in given directory is listed -> Choose the file types to process (i.e.: _png_)
- Select a split strategy.
- Data gets splitted and stored in `train`, `val` and `test` folder in _out_path_.
- Creates an _categories.json_ file in the directory with meta data (classes per split).

#### Contributing
This process can easily be extended by adding a new _xxx_strategy_ method to the __Data Extractor__ module [here](./data_processing/data_creator.py).
1. Add method that name ends with '_strategy'.
2. Implement logic that splits data to a train, validation and test set.

#### Current Implementation
- random_split_strategy: Split the data and balance the classes across the splits.
- preserve_category_strategy: Split the data and preserve a sub category (i.e. signature of one person should be only in train set).

### Train Model

#### Execution
config_file: _training_config_recognition_, _training_config_fraud_, _training_config_detection_
```sh
python main.py -c configs/<config_file>.json run_training
```

#### Config
To run the Model Training for a specific task, first the config file needs to be adjusted:
```json
{
    "encoder": {
        "data": {
            "data_dir": "./data/processed/Petimages/input",
            "set_id": 1,
            "image_dim": [
                299,
                299
            ],
            "train_batch_size": 32,
            "val_batch_size": 4,
            "test_batch_size": 4,
            "category_map": null
        },
        "trainer_args": {
            "logger": true,
            "log_dir": "./results/tb_logs",
            "log_folder": "recognition_encoder",
            "checkpoint_folder": "./results/recognition_encoder",
            "checkpoint_filename": "recognition_encoder-{epoch:02d}-{val_score:.2f}",
            "monitor": "val_score",
            "mode": "max",
            "root_dir": "./results",
            "gpus": 1,
            "check_val_every_n_epoch": 1,
            "max_epochs": 10,
            "profiler": "simple",
            "convert_to_onnx": false
        },
        "hparams": {
            "lr": 0.0001,
            "weight_decay": 0.1,
            "step_size": 10,
            "gamma": 0.1,
            "pretrained": true,
            "image_dim": [
                299,
                299
            ],
            "out_dim": 100,
            "margin": 2.0,
            "train_encoder": true
        }
    },
    "task_head": {
        "data": {
            "data_dir": "./data/processed/Petimages/input",
            "set_id": 0,
            "image_dim": [
                299,
                299
            ],
            "train_batch_size": 16,
            "val_batch_size": 4,
            "test_batch_size": 4,
            "category_map": null
        },
        "trainer_args": {
            "logger": true,
            "log_dir": "./results/tb_logs",
            "log_folder": "recognition_classifier",
            "checkpoint_folder": "./results/recognition_classifier",
            "checkpoint_filename": "recognition_classifier-{epoch:02d}-{val_score:.2f}",
            "monitor": "val_score",
            "mode": "max",
            "root_dir": "./results",
            "gpus": 1,
            "check_val_every_n_epoch": 1,
            "max_epochs": 8,
            "profiler": "simple",
            "convert_to_onnx": false
        },
        "hparams": {
            "weight_decay": 0.001,
            "lr": 0.001,
            "step_size": 2,
            "gamma": 0.1,
            "pretrained": false,
            "image_dim": [
                299,
                299
            ],
            "out_dim": 2,
            "train_encoder": false,
            "head_input_size": 100,
            "task": "classification"
        }
    }
}
```

__data__:  
_data_dir_: Path to the previously created data folder (has to contain train, val and test folder)  
_set_id_: Select Dataset implementation for data.  
_image_dim_: List setting _height_ and _width_ of images.  
_train_batch_size_: Size of training batches.  
_val_batch_size_: Size of validation batches.  
_test_batch_size_: Size of test batches.  
_category_map_: Output if the _preserve_category_strategy_ was used in the data creation step. Else set to `null`.

__trainer_args__:  
_logger_: If `true`, logging is activated. Else set to `false`.  
_log_dir_: Directory to store logs in.  
_log_folder_: Folder in _log_dir_ to save TensorBoard files in.  
_checkpoint_folder_: Directory to store checkpoints in.  
_checkpoint_filename_: Name of the checkpoint files.  
_monitor_: Metric to monitor for checkpoint saving.  
_mode_: Set if metric should be minimized or maximized. Options: `min`/`max`.  
_root_dir_: Root directory for every model output.  
_gpus_: Number of gpus to use, if available. If not set to `0`.  
_check_val_every_n_epoch_: Set validation frequency.  
_max_epochs_: Set maximum number of epochs to train.
_profiler_: Set profiler to 'simple' to obtain time report at the end of training.  
_convert_to_onnx_: If `true`, saves best model as _.onnx_ file.

__hparams__:  
_lr_: Initial learning rate.  
_weight_decay_: Optimizer weight decay rate.  
_step_size_: Frequency to adjust learning rate based on scheduler.  
_gamma_: Scheduler parameter for learning rate adjustment.  
_pretrained_: Use a pre trained ResNet for the encoder.  
_image_dim_: List of hight and width for expected image dimensions.  
_out_dim_: Output length of encoding vector.  
_margin_: Margin for triplet loss function.  
_head_input_size_: Expected length of encoding vector (Has to match the _output_dim_ of the encoder).  
_task_: Task to select an implementation of a task head.  

#### Process
- Execute the main.py using the _training_config_xxx.json_
- Train a task based on the previously anntated images.
- Training on train data set.
- Validation during training on val set. Calculation of val metrics for checkpoint saving.
- Evaluation of pipeline on test set after training based on best checkpoint.

#### Contributing
This process can easily be extended by adding a new [Dataset](./model_training/data_set.py) and/or a new [TaskHead](./model_training/model_module.py):
1. If needed implement a new data loading logic by setting up a new dataset and add the class to the `DATA_SETS` variable [here](./model_training/data_module.py).
2. Implement a new network that predicts a class based on the output vector from the encoder module [here](./model_training/model_module.py).

#### Current Implementation
- Image Recognition: Pipeline and head for image recognition.
- Fraud Analysis: Pipeline and head for fraud analysis.
- Anomaly Detection: Pipeline and head for anomaly detection.

All implementations use a Siamese Network trained with Triplet Loss as the encoder network.