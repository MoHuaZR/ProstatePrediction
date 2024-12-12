# Prostate CT/PET Lymph Node Metastasis Prediction Project

This project aims to predict lymph node metastasis in prostate cancer patients using CT and PET images. The application leverages deep learning models implemented in PyTorch to analyze medical imaging data and provide predictions that can assist in clinical decision-making.

## File Description

- `train.py`: Script for training the deep learning model on prostate CT/PET images.
- `test.py`: Script for testing the trained model on unseen data.
- `setting.py`: Contains project configuration and parameter settings for training and testing.
- `model.py`: Defines the architecture of the deep learning model used for prediction.
- `requirements.txt`: Lists the dependencies of the project.

## Features

- Train and test deep learning models using PyTorch for predicting lymph node metastasis.
- Flexible configuration; users can adjust training parameters in `setting.py`.
- Modular code, making it easy to extend and maintain.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/MoHuaZR/ProstatePrediction.git
   ```
2. Enter the project directory and install the dependencies:
   ```
   cd ProstatePrediction
   pip install -r requirements.txt
   ```

## Dataset

The project uses prostate CT and PET images for training and testing the model. Ensure that you have the dataset available and properly formatted. The dataset should be organized as follows:

```
./data/
  ├── train/
  │     ├── ct/  # CT images for training
  │     ├── pet/ # PET images for training
  └── test/
        ├── ct/  # CT images for testing
        ├── pet/ # PET images for testing
```

The dataset paths should be correctly configured in `setting.py`.

## Usage Instructions

### Train the Model

Use the `train.py` script to train the model on the CT/PET images:
```bash
python train.py
```
The training script will use the deep learning model defined in `model.py` and follow the configuration specified in `setting.py`. Make sure the dataset path and other parameters are set correctly.

In `setting.py`, you can configure the following parameters to control the training process:
- **Learning Rate (`learning_rate`)**: Controls the learning speed of the model, default is `0.001`. Adjust this value based on model convergence.
- **Batch Size (`batch_size`)**: Defines the number of samples per training batch, default is `32`. The choice of batch size affects training speed and stability.
- **Number of Epochs (`epochs`)**: Specifies the number of times to iterate over the entire dataset, default is `10`. Increasing the number of epochs can improve model accuracy but also increases training time.
- **Dataset Path (`data_path`)**: Specifies the location of the training dataset. Make sure the dataset is downloaded and placed at the specified path.

For example, you can set the following in `setting.py`:
```python
learning_rate = 0.001
batch_size = 64
epochs = 20
data_path = "./data/train"
```
Then run the training script to start the training process.

### Test the Model

Use the `test.py` script to test the model:
```bash
python test.py
```
The testing script will load the trained model and evaluate it on the test dataset. Users can adjust testing parameters, such as the test dataset path and batch size, in `setting.py`.

In `setting.py`, you can set:
- **Test Dataset Path (`test_data_path`)**: Specifies the location of the test dataset.
- **Test Batch Size (`test_batch_size`)**: Defines the number of samples per test batch.

For example:
```python
test_data_path = "./data/test"
test_batch_size = 32
```
Then run the test script to evaluate the model's performance.

### Parameter Configuration

You can configure the training and testing parameters in `setting.py`, such as:
- Learning Rate (`learning_rate`)
- Batch Size (`batch_size`)
- Number of Epochs (`epochs`)
- Training Dataset Path (`data_path`)
- Test Dataset Path (`test_data_path`)
- Test Batch Size (`test_batch_size`)

## Dependencies

This project depends on the following Python libraries, listed in `requirements.txt`:

```
pip>=9.0.1
torch==0.4.1
numpy==1.15.4
nibabel==2.4.1
scipy==1.1.0
argparse==1.1
```

## Contributing

Contributions to this project are welcome. Feel free to submit suggestions or PRs to help improve the code and functionality.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it.
