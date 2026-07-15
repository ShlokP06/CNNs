# Neural Networks

Two practical CNN/GAN implementations built around a real-vs-AI image
dataset.

## Tasks

**Task 1 — Real vs. AI image classifier**
A supervised CNN trained on labeled `Train/Real` and `Train/AI` folders to
distinguish real photos from AI-generated ones. Images are normalized and
resized, the model is trained and evaluated on a held-out test set, and
predictions are written out as `(image name, label)` pairs to a CSV.

**Task 2 — GAN for AI image generation**
A generator/discriminator pair trained adversarially to produce realistic
AI-generated images, with generated samples checked at regular intervals
during training. Uses a Kaggle landscape dataset
(`utkarshsaxenadn/landscape-recognition-image-dataset-12k-images`) for
training data.

## Setup

```bash
git clone https://github.com/ShlokP06/CNNs.git
cd CNNs
pip install numpy pandas matplotlib tqdm tensorflow torch scikit-learn
```

Data is expected as `Train/` (with `Real/` and `AI/` subfolders) and
`Test/` folders; if provided as zip archives, unzip before running.

For Task 2's Kaggle dataset:

```bash
pip install kaggle
# place your kaggle.json API key in ~/.kaggle/
kaggle datasets download utkarshsaxenadn/landscape-recognition-image-dataset-12k-images
unzip landscape-recognition-image-dataset-12k-images.zip
```

## Stack

`TensorFlow/Keras` · `PyTorch` · `scikit-learn` · `NumPy` · `Pandas` ·
`Matplotlib`
