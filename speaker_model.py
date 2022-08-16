import numpy as np
from speechbrain.pretrained import EncoderClassifier
from keras.models import load_model
import tensorflow as tf

import torch

# standard_scaler_dict_path = r"C:/git/x_vectors_embedding/standard_scaler_dict.npy"
# speaker_model_path = r"C:/git/x_vectors_embedding/waveform_to_speaker_emb.h5"

# standard_scaler_dict_path = r"C:/git/x_vectors_embedding/standard_scaler_dict_orig.npy"
# speaker_model_path = r"C:/git/x_vectors_embedding/waveform_to_speaker_emb_mod_small.h5"

standard_scaler_dict_path = r"src/speaker_model/standard_scaler_dict_orig.npy"
speaker_model_path = r"src/speaker_model/waveform_to_speaker_emb_mod_small.h5"

class SpeakerModel:
    def __init__(self):
        print("Load Speaker model")
        standard_scaler_dict = np.load(standard_scaler_dict_path, allow_pickle=True).item()
        self.mean = standard_scaler_dict['mean']
        self.scale = standard_scaler_dict['scale']
        self.base_x_vectors = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
        with tf.device('/cpu:0'):
            self.speaker_model = load_model(speaker_model_path)

    def _norm_standard(self, features):
        return (features - self.mean) / self.scale

    def __call__(self, waveform):
        features = self.base_x_vectors.encode_batch(torch.Tensor(waveform)).squeeze()
        # features = self.base_x_vectors.encode_batch(waveform).squeeze()
        norm_features = self._norm_standard(features)
        # speaker_id = np.argmax(self.speaker_model.predict(norm_features.numpy()),axis=0)
        with tf.device('/cpu:0'):
            # speaker_id = self.speaker_model.predict(norm_features.numpy())
            # speaker_id = self.speaker_model.predict(norm_features.unsqueeze(0).numpy())
            if norm_features.dim() == 1:
                speaker_id = self.speaker_model.predict(norm_features.unsqueeze(0).numpy())
            else:
                speaker_id = self.speaker_model.predict(norm_features.numpy())
        return speaker_id



if __name__ == "__main__":
    sm = SpeakerModel()
    import librosa
    import pandas as pd

    speakers_data_path = r"src/speakers_data.csv"
    speakers_data = pd.read_csv(speakers_data_path).set_index('Id')
    speakers_data_dict = speakers_data.to_dict('index')
    speakers_list = list(speakers_data_dict.keys())



    audio_path_p225 = r"src/demo/p225_023_mic2_gt.wav"
    waveform, Librosa_sample_rate = librosa.load(audio_path_p225)
    #input = torch.rand(16, 8960)
    print(sm(waveform))
    output_speaker = speakers_list[np.argmax(sm(torch.Tensor(waveform).unsqueeze(0)))]


    print(output_speaker)
