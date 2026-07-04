# Messy Mashup: Music Genre Classification

Hey there! This is our Deep Learning / GenAI project where we classify 30-second audio clips into 10 different music genres: **blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock**. 

The catch? The audio files are "messy mashups"—they have overlapping vocals, drums, bass, and instrument stems, mixed with heavy environmental background noise. We had to build a robust audio processing and modeling pipeline to figure out the correct genre under these noisy conditions.

---

## 🚀 How Our Pipeline Works

### 1. Data Augmentation & Stem Mixing
Because real-world music has clean parts and noisy parts, we didn't train on simple tracks. We built a custom training dataset generator that:
* **Mixes stems on the fly:** We randomly select stems (`vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`) from different songs of the same genre and mix them.
* **Adds environmental noise:** We inject random noise profiles from the ESC-50 dataset at varying Signal-to-Noise Ratios (SNR).
* **Augments the audio:** We apply random time shifting, cropping (to 512 frames), volume scaling, and SpecAugment (frequency and time masking) on the Mel-Spectrogram.

### 2. Audio Processing
We used `librosa` and `torchaudio` to process the waveforms:
* Sample Rate: Resampled to **22,050 Hz**
* Representation: **Mel-Spectrogram** (128 Mel bands, FFT size 2048, hop length 512)
* Normalization: Subtracted the mean and divided by standard deviation to scale features.

---

## 🧠 Models We Explored

We tried a few different architectures to see what works best:

| Model | Architecture Details | Validation Accuracy / F1 |
| --- | --- | --- |
| **LightGBM (Baseline)** | Classical ML model trained on MFCC feature extractions. | ~90.0% / 0.895 F1 |
| **CRNN (CNN + LSTM)** | Custom 7-layer CNN feature extractor combined with a Bidirectional LSTM to model temporal patterns. | ~90.2% / 0.901 F1 |
| **AST (Audio Spectrogram Transformer)** | A Transformer model (`MIT/ast-finetuned-audioset-10-10-0.4593`) fine-tuned on our custom augmented data. | **~93.0%** (Best) |

Our fine-tuned AST model is uploaded on Hugging Face at [udit789/Messy-Mashup-AST](https://huggingface.co/udit789/Messy-Mashup-AST).

---

## 📁 Repository Structure

Here’s a quick guide to what is in each folder:

* 📂 [deployment/](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/deployment): Contains files to deploy the model as a web app.
  * [app.py](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/deployment/app.py): The main web app file using Gradio.
  * [requirements.txt](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/deployment/requirements.txt): List of libraries needed to run the web app.
* 📂 [notebooks/](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/notebooks): Implementation of our different models.
  * [classical-ml.ipynb](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/notebooks/classical-ml.ipynb): Classical ML experiments using LightGBM.
  * [rcnn-model3.ipynb](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/notebooks/rcnn-model3.ipynb): CNN + LSTM neural net training code.
  * [audio-spectrogram-transformer-pretrained-model.ipynb](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/notebooks/audio-spectrogram-transformer-pretrained-model.ipynb): AST fine-tuning code.
* 📂 [Milestone/](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/Milestone): Standard project progress notebooks (Milestones 1 to 5).
* 📂 [EDA/](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/EDA): Exploratory Data Analysis of the audio files and stems.
* 📂 [learning/](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/learning): Notebooks to learn about Mel-spectrogram generation and audio tasks.
* 📄 [DL-23f2002974-notebook-t12026.ipynb](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/DL-23f2002974-notebook-t12026.ipynb): Main training notebook showing preprocessing, architecture setups, and training loops.
* 📄 [inference-notebook.ipynb](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/inference-notebook.ipynb): Notebook used to generate Kaggle test predictions using our fine-tuned AST model.
* 📄 [Project Report.pdf](file:///Users/udit/Desktop/DLgenai%20-%20Project/DL-23f2002974-notebook-t12026/Project Report.pdf): Detailed report explaining our findings, math tasks, and training progress.

---

## 🛠️ How to Run the Web App Local Server

We built a simple Gradio UI to test the classifier locally.

### Step 1: Install dependencies
Make sure you have python installed. Run this command to install the required libraries:
```bash
pip install -r deployment/requirements.txt
```

### Step 2: Run the app
Start the local server using:
```bash
python deployment/app.py
```
This will download our fine-tuned AST model weights (`finetuned-AST.pth`) automatically from Hugging Face and launch a local web link (usually `http://127.0.0.1:7860`). Open that link in your browser, drag and drop any audio clip, and watch it predict the genre!

---

## 🎓 Project Takeaways
1. **SpecAugment & Noise Augmentation work wonders:** Training on noisy mixed stems forced our models to learn actual musical features instead of just memorizing quiet backgrounds.
2. **Transformers rule audio too:** The Audio Spectrogram Transformer (AST) easily beat standard CNNs and classical ML baselines because it captures global context over time-frequency dimensions much better.