import numpy as np
import librosa
from keras.models import load_model
import tensorflow as tf

import torch

standard_scaler_dict_path = r"C:/git/Paccent_classifier/accent_scaler_dict_torch.npy"
id_to_accent_mapping_path = r"C:/git/Paccent_classifier/id_to_accent_mapping.npy"
accent_model_path = r"C:/git/Paccent_classifier/waveform_to_accent_emb_wo_norm.h5"


class AccentModel:
    def __init__(self):
        standard_scaler_dict = np.load(standard_scaler_dict_path, allow_pickle=True).item()
        self.mean = standard_scaler_dict['mean']
        self.scale = standard_scaler_dict['scale']
        self.sample_rate = 16_000
        self.num_mfcc = 20
        self.id_to_accent_mapping = np.load(id_to_accent_mapping_path, allow_pickle=True).item()
        with tf.device('/cpu:0'):
            self.accent_model = load_model(accent_model_path)

    def get_accent_mapping(self):
        return self.id_to_accent_mapping

    def _norm_standard(self, features):
        # return ((features.T - self.mean) / self.scale).T
        return (features - self.mean) / self.scale

    def __call__(self, waveform):
        mfcc = [librosa.feature.mfcc(y=one_waveform.numpy(), sr=self.sample_rate, n_mfcc=self.num_mfcc) for one_waveform
                in waveform]
        mfcc_mean = [[np.mean(i.T, axis=0) for i in one_mfcc] for one_mfcc in mfcc]
        norm_features = self._norm_standard(mfcc_mean)
        # speaker_id = np.argmax(self.speaker_model.predict(norm_features.numpy()),axis=0)
        with tf.device('/cpu:0'):
            accent_pred = self.accent_model.predict(norm_features)
            # accent_predself.accent_model.predict(np.expand_dims(norm_features, axis=0))

        return accent_pred


if __name__ == "__main__":
    am = AccentModel()
    input = torch.rand(16, 8960)
    print(am(input))
    print(am.get_accent_mapping())
    print("Bammmm")
