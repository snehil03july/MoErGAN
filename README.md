# MoErGAN: Motion Error Generative Adversarial Network for MRI

## Introduction
MoErGAN is an innovative solution designed to address the scarcity of motion-corrupted MRI data for the development of advanced motion correction algorithms. By synthesizing realistic motion artifacts in error-free MRI images, MoErGAN facilitates the augmentation of datasets, enabling robust training and improvement of deep learning-based motion correction techniques.

![MoErGAN Overview](<path/to/overview_image.png>)

## Features
- **Data Augmentation**: Generates high-quality motion-corrupted MRI data from error-free scans.
- **Deep Learning Model**: Utilizes a state-of-the-art Generative Adversarial Network (GAN) architecture.
- **Flexible and Scalable**: Supports various MRI modalities and is scalable to large datasets.
- **Open Source**: Enables contributions and use by the research and medical communities.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Python 3.8+
- Pip or Conda
- Access to GPU (recommended for training)

### Installation
1. Clone the repository: git clone https://github.com/snehil03july/MoErGAN.git
2. Navigate to the project directory: cd MoErGAN
3. Install required Python packages: pip install requirements.txt

## Usage
Follow these steps to run MoErGAN for generating motion-corrupted MRI images:

1. Prepare your dataset according to the instructions in `<path/to/data_preparation_instructions.md>`.
2. Train the MoErGAN model with your dataset: python train.py --dataset <path/to/your/dataset>
3. Generate motion-corrupted MRI images: python generate.py --model <path/to/trained/model> --output <path/to/output/images>

# Flask GAN App

This Flask application allows you to upload a NIfTI (.nii or .nii.gz) image file, processes it using a pre-trained U-Net GAN model, and then displays the generated output image.

## Project Structure

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. **Clone the Repository and Navigate to the Project Directory**

   If you haven't already, clone the repository and navigate to the project directory:

   ```bash
   git https://github.com/snehil03july/MoErGAN.git
   cd MoErGAN
## File Structure
/flask_gan_app
│
├── app.py                        # Main Flask application
├── unet_generator.pth            # Trained model weights
│
├── /uploads                      # Directory to store uploaded NIfTI files
│   └── (uploaded files will be saved here)
│
├── /static                       # Static files directory
│   └── /generated                # Directory to store generated images
│
└── /templates                    # HTML template files
    ├── index.html                # File upload page
    └── display.html              # Display page for generated images
run - python app.py
host - http://127.0.0.1:5000/

## Documentation
For comprehensive documentation, including the model architecture, training process, and API references, please refer to the `/docs` directory.

## Contributing
We welcome contributions from the community, including bug reports, feature requests, and code submissions. Please see `CONTRIBUTING.md` for more details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Gratitude to all contributors and the open-source community.

## Contact
For any inquiries, please reach out to `sk895@exeter.ac.uk`.

## Project Status
This project is currently in `<development>` phase. For the latest updates, please check the project's GitHub repository.





