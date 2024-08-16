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
## M-Art Dataset

The M-Art (Motion-Artifacts) dataset is a collection of MRI scans specifically curated to study and mitigate the effects of motion artifacts in brain imaging. This dataset includes MRI scans captured under varying conditions of head motion: no motion, slight motion, and extreme motion.

### Dataset Access

The M-Art dataset can be accessed and downloaded from the following sources:

- **OpenNeuro**: The dataset is publicly available on OpenNeuro at [M-Art Dataset on OpenNeuro](https://openneuro.org/datasets/ds004173/versions/1.0.2).
  
- **University of Exeter OneDrive**: For members of the University of Exeter, the dataset is also available via OneDrive. Please use your university email ID to access the dataset: [M-Art Dataset on OneDrive](https://universityofexeteruk-my.sharepoint.com/:f:/g/personal/sk895_exeter_ac_uk/EiAEft9oryhAhJXbj6lJJGsBD_HwZK5J7WF0QcxLlNwOfg?e=4zsWv3).

### Data Contents

The dataset includes the following:

- **No Motion**: MRI scans captured with no intentional head motion.
- **Slight Motion**: MRI scans captured with slight, controlled head motion.
- **Extreme Motion**: MRI scans captured with significant head motion.

This dataset is ideal for research in motion artifact correction, training machine learning models, and enhancing the quality of MRI images.

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





