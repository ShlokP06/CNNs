
# Neural Networks
A couple of scripts targeted at basic implementation of Neural Networks for real life usage.

## Table of Contents

- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Tasks](#tasks)
  - [Task 1: Real vs AI Image Classifier](#task-1-real-vs-ai-image-classifier)
  - [Task 2: GAN Model for AI Images](#task-2-gan-model-for-ai-images)
- [Usage](#usage)
- [Contact](#contact)

---

## About the Project

This revolved around images - how the infamous modern technology deepfake works, how can we use it to generate AI images as well as how AI itself can differentiate between a real and a fake image.

The total project had 2 tasks:

- A Real vs AI image classifier
- Training a GAN model to generate AI images

---

## Getting Started

### Prerequisites libraries

The following libraries played an important role in making this project work:

- TensorFlow Keras
- NumPy
- Pandas
- TQDM
- OS
- scikit-learn
- PyTorch
- Matplotlib

### Installation

Step-by-step guide to installing and running your project.

1. Clone the repository:

   ```bash
   git clone https://github.com/ShlokP06/CynapticsInduction.git
   ```

   [https://github.com/ShlokP06/CynapticsInduction](https://github.com/ShlokP06/CynapticsInduction)

2. Navigate to the project directory:

   ```bash
   cd CynapticsInduction
   ```

3. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib tqdm tensorflow pytorch sklearn
   ```
---

## Tasks

### Task 1: Real vs AI Image Classifier

This subtask involves creating a classifier to distinguish between real and AI-generated images. The model uses a supervised learning approach and the labeled datasets in the `Train/Real` and `Train/AI` folders.

Steps to complete:

1. Preprocess the data by normalizing and resizing the images.
2. Train a convolutional neural network (CNN) on the training dataset.
3. Evaluate the model's performance using the test dataset.
4. Writing the output in the form of Image Name, Label in a .csv file.

#### Uploading Data

Instructions for uploading data to the project:

1. The project operates on two folders : Train and Test. The Train folder again is subdivided into 2 folders: Real and AI containing the respective images for training the model.

2. Ensure your data files are properly formatted and stored as image files.

3. Place the data files in the designated directory within the project folder.

4. Update the configuration file (if applicable) to specify the paths to your data files.

5. If the folders are uploaded as zip archives, try running:

   ```bash
   unzip Train.zip
   unzip Test.zip
   ```

6. Verify that the data has been successfully loaded by checking the logs or output files.

### Task 2: GAN Model for AI Images

This subtask focuses on training a Generative Adversarial Network (GAN) to generate realistic AI images. The GAN model includes a generator and a discriminator network.

Steps to complete:

1. Preprocess the training images to prepare them for input to the GAN.
2. Train the generator and discriminator alternately using the specified loss functions.
3. Displaying the images at a regular interval to check the proper functioning of the system.
4. Evaluate the quality of generated images and fine-tune the model.

#### Importing Kaggle Database

I had to incorporate a Kaggle dataset for training the GAN model:

1. Install the Kaggle Python package:

   ```bash
   pip install kaggle
   ```

2. Obtain your Kaggle API key by visiting your Kaggle account settings and downloading the `kaggle.json` file.

3. Place the `kaggle.json` file in the appropriate directory (e.g., `~/.kaggle/`):

   ```bash
   mkdir ~/.kaggle
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. Use the Kaggle CLI to download the dataset:

   ```bash
   kaggle datasets download utkarshsaxenadn/landscape-recognition-image-dataset-12k-images
   ```

5. Unzip the downloaded dataset:

   ```bash
   unzip landscape-recognition-image-dataset-12k-images.zip
   ```

6. Update the training script to use the downloaded dataset.

---

## Usage

Instructions and examples on how to use the project. Optionally, include screenshots or code examples:

---

## Contact

Shlok Parikh - [shlokparikhchoco@gmail.com](mailto\:shlokparikhchoco@gmail.com)

Project Link: [https://github.com/ShlokP06/CynapticsInduction](https://github.com/ShlokP06/CynapticsInduction)
