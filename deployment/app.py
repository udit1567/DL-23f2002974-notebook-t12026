import torch
import librosa
import numpy as np
import gradio as gr
from transformers import ASTForAudioClassification
from huggingface_hub import hf_hub_download

#DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MODEL
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

model = ASTForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=10,
    ignore_mismatched_sizes=True
)

#DOWNLOAD MODEL FROM HUGGINGFACE
model_path = hf_hub_download(
    repo_id="udit789/Messy-Mashup-AST",
    filename="finetuned-AST.pth"
)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

#LABELS
genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

#AUDIO SETTINGS
target_w = 1024
target_len = 22050 * 30


#AUDIO PREPROCESS
def preprocess_audio(audio_path):

    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    mel_t = mel_db.T

    if mel_t.shape[0] > target_w:
        start = (mel_t.shape[0] - target_w) // 2
        mel_final = mel_t[start:start + target_w]
    else:
        mel_final = np.pad(mel_t, ((0, target_w - mel_t.shape[0]), (0, 0)))

    mel_tensor = torch.tensor(mel_final, dtype=torch.float32)

    mel_tensor = mel_tensor.unsqueeze(0).to(device)

    return mel_tensor


#PREDICTION
def predict(audio):

    mel = preprocess_audio(audio)

    with torch.no_grad():

        outputs = model(mel)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    result = {}

    for g, p in zip(genres, probs):
        result[g] = float(p)

    return result


#GRADIO UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label(num_top_classes=10),
    title="Audio Genre Classifier using AST (finetuned)",
    description="Upload an audio clip and the model predicts the music genre."
)

demo.launch()