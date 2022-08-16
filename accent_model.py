import numpy as np
import librosa
from keras.models import load_model
import tensorflow as tf
from speechbrain.pretrained import EncoderClassifier


import torch

Mfcc_model = False #or x-vectors

if Mfcc_model:
    # mfcc model
    standard_scaler_dict_path = r"C:/git/Paccent_classifier/accent_scaler_dict_torch.npy"
    accent_model_path = r"C:/git/Paccent_classifier/waveform_to_accent_emb_wo_norm.h5"
else:
    # x-vectors model
    # standard_scaler_dict_path = r"C:/git/Paccent_classifier/accent_scaler_dict_torch_accent.npy"
    # accent_model_path = r"C:/git/Paccent_classifier/x_vector_to_accent_with_64_dim_same_arch.h5"
    standard_scaler_dict_path = r"src/accent_model/accent_scaler_dict_torch_accent.npy"
    accent_model_path = r"src/accent_model/x_vector_to_accent_with_64_dim_same_arch_12_classes.h5"

# id_to_accent_mapping_path = r"C:/git/Paccent_classifier/id_to_accent_mapping.npy"
id_to_accent_mapping_path = r"src/accent_model/id_to_accent_mapping.npy"


class AccentModel:
    def __init__(self):
        print("Load Accent model")
        standard_scaler_dict = np.load(standard_scaler_dict_path, allow_pickle=True).item()
        self.mean = standard_scaler_dict['mean']
        self.scale = standard_scaler_dict['scale']
        self.sample_rate = 16_000

        if Mfcc_model:
            self.num_mfcc = 20
        else:
            self.base_x_vectors = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                                 savedir="pretrained_models/spkrec-xvect-voxceleb")

        self.id_to_accent_mapping = np.load(id_to_accent_mapping_path, allow_pickle=True).item()

        with tf.device('/cpu:0'):
            self.accent_model = load_model(accent_model_path)

    def get_accent_mapping(self):
        return self.id_to_accent_mapping

    def _norm_standard(self, features):
        # return ((features.T - self.mean) / self.scale).T
        return (features - self.mean) / self.scale

    def __call__(self, waveform):
        if Mfcc_model:
            mfcc = [librosa.feature.mfcc(y=one_waveform.numpy(), sr=self.sample_rate, n_mfcc=self.num_mfcc) for one_waveform
                    in waveform]
            features = [[np.mean(i.T, axis=0) for i in one_mfcc] for one_mfcc in mfcc]
        else:
            features = self.base_x_vectors.encode_batch(torch.Tensor(waveform)).squeeze()

        norm_features = self._norm_standard(features)
        # speaker_id = np.argmax(self.speaker_model.predict(norm_features.numpy()),axis=0)
        with tf.device('/cpu:0'):
            if Mfcc_model:
                accent_pred = self.accent_model.predict(norm_features)
            else:
                if norm_features.dim() == 1:
                    accent_pred = self.accent_model.predict(norm_features.unsqueeze(0).numpy())
                else:
                    accent_pred = self.accent_model.predict(norm_features.numpy())
            # accent_predself.accent_model.predict(np.expand_dims(norm_features, axis=0))

        return accent_pred


if __name__ == "__main__":
    am = AccentModel()
    input = torch.rand(16, 8960)
    print(am(input))
    print(am.get_accent_mapping())
    print("Bammmm")
