import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2Model

class HuggingFaceFeatureExtractor:
    def __init__(self, model_class, name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name)
        self.model = model_class.from_pretrained(name, output_hidden_states=True)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.hidden_states

FEATURE_EXTRACTORS = {
    "wav2vec2-xls-r-2b": lambda: HuggingFaceFeatureExtractor(
        Wav2Vec2Model, "facebook/wav2vec2-xls-r-2b"
    ),
}

def read_metadata(file_path):
    relevant_files = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                relevant_files.append(parts[0])
    return sorted(relevant_files)


def main(outdir, metadata_file):
    relevant_files = read_metadata(metadata_file)
    print(f"Metadata contains {len(relevant_files)} files.")
    feature_extractor = FEATURE_EXTRACTORS['wav2vec2-xls-r-2b']()

    layer_embeddings = [[] for _ in range(48)]

    for fi in tqdm(relevant_files):
        if os.path.exists(fi):
            audio, sr = librosa.load(fi, sr=16000)
            hidden_states = feature_extractor(audio, sr)
            for layer_idx in range(48):
                layer_output = hidden_states[layer_idx]
                ## average pooling on time frames
                mean_layer_output = torch.mean(layer_output, dim=1).cpu().numpy()
                layer_embeddings[layer_idx].append(mean_layer_output)

    for layer_idx in range(48):
        stacked_embeddings = np.vstack(layer_embeddings[layer_idx])
        np.save(os.path.join(outdir, f'wav2vec2-xls-r-2b_Layer{layer_idx}_for.npy'), stacked_embeddings)

if __name__ == '__main__':
    print('script running')
    outdir = './feats/wav2vec2-xls-r-2b'
    metadata_file = './processed_metadata/for_systems.csv'
    main(outdir, metadata_file)

