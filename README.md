# CBD-bone: Bridging the Representational Gap for Trustworthy Skeletal Segmentation

This repository contains the official implementation of **CBD-bone**, a frequency-spatial synergistic framework designed for high-fidelity skeletal segmentation in CT imaging.

## 1. Project Structure

- `new_model_structure/`: Core architectural components.
  - `FFC`: Implementation of Fast Fourier Convolution for global frequency perception.
  - `CPCA3dNew1`: Contextual Pixel Correlation Attention (CPCA) for local refinement.
  - `Unet_Model_updata`: The main CBD-bone architecture combining SCB and SDB modules.
- `poas`: Implementation of Patch-Occlusion Attribution for Segmentation (POAS), our novel XAI method.
- `config.py`: Hyperparameters and configuration settings for training and inference.
- `train_new2_sj.py`: Main script for model training and validation.
- `README.md`: This file.
- `LICENSE`: MIT License.

## 2. Environment Setup

The code is implemented in Python 3.9+ and PyTorch. To install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install monai nibabel numpy pandas
