# Deep Learning Architectures Hub

A comprehensive repository showcasing implementations of various deep learning neural network architectures using Python. This hub provides detailed implementations and explanations of modern and classical neural network models used in computer vision, natural language processing, audio processing, and more.

## 📋 Table of Contents

- [Overview](#overview)
- [Architectures](#architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Architecture Categories](#architecture-categories)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository is dedicated to providing clear, well-documented implementations of deep learning architectures that have shaped the field of artificial intelligence. Each architecture is implemented from scratch or using foundational libraries, with detailed comments explaining the core concepts and design decisions.

### Key Features

✅ Comprehensive implementations of 33+ neural network architectures  
✅ Well-organized directory structure with one architecture per folder  
✅ Detailed documentation and usage examples  
✅ Python-based implementations  
✅ Educational focus with explanatory comments  
✅ Multiple application domains covered  

---

## Architectures

### Convolutional Neural Networks (CNNs)

#### Classic CNN Architectures
- **[AlexNet](./alex-net/)** - Pioneering deep CNN that won ImageNet 2012, featuring multiple convolutional and pooling layers
- **[LeNet](./le-net/)** - Foundational CNN architecture for handwritten digit recognition
- **[ZFNet](./zf-net/)** - Improved version of AlexNet with better visualization and understanding
- **[ResNet](./res-net/)** - Residual Networks enabling very deep architectures (50, 101, 152 layers)
- **[VGG-Net](./v-net/)** - Simple yet powerful architecture demonstrating the importance of network depth

#### Efficient CNN Architectures
- **[MobileNet](./mobile-net/)** - Lightweight CNN designed for mobile and edge devices
- **[SqueezeNet](./squeeze-net/)** - Compact architecture with AlexNet-level accuracy using fewer parameters
- **[ShapeNet](./shape-net/)** - Specialized for 3D shape classification and analysis

#### Advanced CNN Variants
- **[PointNet](./point-net/)** - Direct learning from point clouds without voxelization
- **[SegNet](./seg-net/)** - Semantic segmentation architecture with encoder-decoder structure
- **[CoAtNet](./coAt-net/)** - Hybrid architecture combining convolutions and self-attention

### Recurrent Neural Networks (RNNs)

- **[Recurrent Networks](./recurrent-net/)** - LSTM, GRU, and vanilla RNN implementations for sequence modeling
- **[Liquid State Machines](./lquid-net/)** - Spiking neural networks for temporal sequence processing

### Transformer & Attention-Based Architectures

- **[RetNet](./ret-net/)** - Retention-based alternative to Transformers
- **[XLNet](./xl-net/)** - Autoregressive language model with permutation language modeling
- **[Tacotron](./tacotron/)** - Text-to-speech synthesis using attention mechanisms

### Specialized Architectures

#### Audio & Signal Processing
- **[WaveNet](./wave-net/)** - Generative model for raw audio waveforms
- **[Tacotron](./tacotron/)** - End-to-end text-to-speech synthesis

#### 3D & Volumetric Data
- **[VoxNet](./vox-net/)** - 3D convolutional network for volumetric data
- **[3D Segmentation (V-Net)](./v-net/)** - 3D medical image segmentation
- **[PointNet](./point-net/)** - Point cloud processing and classification

#### Reinforcement Learning
- **[DQN (Deep Q-Network)](./deepq-net/)** - Deep reinforcement learning for game playing
- **[Deep Observation](./deepo-net/)** - Advanced observations in deep learning

#### Specialized Domains
- **[Physics-Informed Neural Networks](./physics-informed-net/)** - Neural networks incorporating physics constraints
- **[DeepInterest Networks](./deepinterest-net/)** - Networks for interest prediction and recommendation
- **[TabNet](./tab-net/)** - Specialized architecture for tabular/structured data
- **[BitNet](./bit-net/)** - Binary neural networks for extreme efficiency
- **[YAMNet](./yam-net/)** - Audio event detection and classification

#### Generative Models
- **[Capsule Networks](./capsule-net/)** - Novel architecture with capsules for part-whole relationships
- **[DAN-Net (Deep Attention Network)](./dan-net/)** - Networks leveraging attention mechanisms
- **[Flow-Net](./flow-net/)** - Optical flow estimation networks

#### Forecasting & Time Series
- **[FourCastNet](./fourcast-net/)** - Weather forecasting using deep learning

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/krishnashashanth-sks/deeplearning-architectures-hub.git
   cd deeplearning-architectures-hub
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Common dependencies typically include:
   - `tensorflow` or `pytorch` - Deep learning frameworks
   - `numpy` - Numerical computing
   - `matplotlib` - Visualization
   - `scipy` - Scientific computing

---

## Usage

### Running Individual Architectures

Each architecture folder is self-contained and can be run independently. Navigate to the specific architecture folder:

```bash
cd alex-net
python alexnet.py
```

### Example: Training a Model

```python
from architectures.alexnet import AlexNet
import torch

# Initialize model
model = AlexNet(num_classes=10)

# Forward pass
input_data = torch.randn(32, 3, 224, 224)
output = model(input_data)

print(f"Output shape: {output.shape}")  # torch.Size([32, 10])
```

### Jupyter Notebooks

Each architecture folder typically includes:
- Implementation files (e.g., `alexnet.py`)
- Notebook tutorials (e.g., `tutorial.ipynb`)
- Example usage scripts
- Test cases

---

## Repository Structure

```
deeplearning-architectures-hub/
├── README.md
├── requirements.txt
│
├── alex-net/                    # AlexNet implementation
├── le-net/                      # LeNet for digit recognition
├── v-net/                       # VGG-style networks
├── res-net/                     # ResNet with skip connections
├── zf-net/                      # Zeiler & Fergus Net
│
├── mobile-net/                  # Mobile-optimized CNN
├── squeeze-net/                 # SqueezeNet (compact)
├── capsule-net/                 # Capsule Networks
│
├── point-net/                   # Point Cloud Processing
├── seg-net/                     # Semantic Segmentation
├── shape-net/                   # 3D Shape Analysis
│
├── recurrent-net/               # RNN/LSTM implementations
├── lquid-net/                   # Liquid State Machines
│
├── wave-net/                    # Audio generation
├── tacotron/                    # Text-to-speech
├── yam-net/                     # Audio event detection
│
├── xl-net/                      # XLNet language model
├── ret-net/                     # Retention Networks
│
├── deepq-net/                   # Deep Q-Learning
├── deepinterest-net/            # Interest prediction
├── deepo-net/                   # Deep observations
│
├── tab-net/                     # Tabular data
├── bit-net/                     # Binary networks
├── dan-net/                     # Attention networks
│
├── flow-net/                    # Optical flow
├── coAt-net/                    # Convolutional + Attention
├── vox-net/                     # 3D volumetric data
│
├── physics-informed-net/        # Physics-informed models
├── fourcast-net/                # Weather forecasting
│
└── ... (additional architectures)
```

---

## Architecture Categories

### 📷 **Computer Vision**
Architectures for image classification, object detection, and semantic segmentation:
- AlexNet, LeNet, ResNet, VGGNet, MobileNet, SqueezeNet
- SegNet, CoAtNet, ShapeNet, ZFNet

### 🎤 **Audio & Speech Processing**
Models for audio generation, synthesis, and analysis:
- WaveNet, Tacotron, YAMNet

### 🔢 **Sequential & Time Series**
Networks for processing temporal data:
- Recurrent Networks, Liquid State Machines, RetNet, XLNet

### 🎮 **Reinforcement Learning**
Deep learning for decision-making:
- DQN (Deep Q-Networks)

### 📊 **Specialized Data Types**
- **3D/Volumetric:** PointNet, VoxNet, V-Net
- **Tabular:** TabNet
- **Binary:** BitNet

### 🔬 **Advanced & Emerging**
Cutting-edge architectures:
- Physics-Informed Neural Networks, CapsuleNets, Flow-Net, FourCastNet

---

## Getting Started Guide

1. **Choose an architecture** from the list above
2. **Navigate to its folder:** `cd architecture-name`
3. **Read the documentation** in the folder's README
4. **Install required packages** (if any are specific to that architecture)
5. **Run the example scripts** or notebooks
6. **Experiment and modify** the code to understand the architecture better

---

## Key Concepts Covered

- Convolutional operations and pooling
- Residual connections and skip connections
- Attention mechanisms
- Recurrent architectures (LSTM, GRU)
- Generative models
- Transfer learning techniques
- Batch normalization and regularization
- Optimization strategies (SGD, Adam, etc.)

---

## Recommended Learning Path

### Beginner
1. LeNet → Classic CNN fundamentals
2. AlexNet → Deep learning breakthrough
3. VGGNet → Understanding depth in networks
4. ResNet → Skip connections and very deep networks

### Intermediate
1. MobileNet → Efficient architectures
2. Recurrent Networks → Sequence modeling
3. Attention Mechanisms → Modern approaches
4. SegNet → Semantic segmentation

### Advanced
1. CapsuleNets → Novel architectural paradigms
2. Physics-Informed Networks → Domain-specific integration
3. XLNet → Advanced language modeling
4. PointNet → 3D data processing

---

## Contributing

Contributions are welcome! If you'd like to add a new architecture or improve existing implementations:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-architecture`)
3. Add your implementation with:
   - Well-commented code
   - Documentation explaining the architecture
   - Usage examples
   - Test cases
4. Commit your changes (`git commit -am 'Add new architecture'`)
5. Push to the branch (`git push origin feature/new-architecture`)
6. Open a Pull Request

---

## Requirements

See `requirements.txt` for the full list of dependencies. Common frameworks used:

- **Deep Learning:** TensorFlow, PyTorch
- **Scientific Computing:** NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Data Processing:** Pandas, Scikit-learn

---

## License

This project is open source and available for educational and research purposes.

---

## References & Resources

- [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

---

## Repository Information

- **Created:** March 27, 2026
- **Language:** Python
- **Status:** Active Development
- **Author:** krishnashashanth-sks

---

## Support

For questions, issues, or suggestions:
- Open an GitHub issue in the repository
- Check existing documentation in individual architecture folders
- Review research papers linked in the References section

---

**Happy Learning! 🚀**

Feel free to explore, learn, and build amazing things with deep learning!
