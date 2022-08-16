import joblib
import torch
import soundfile as sf

# change in modeling_hubert.pt line 681  for layer in self.layer =>  for layer in self.layer[:6]
from transformers import HubertModel


kmeans_model_path_global = "C:/Users/User/Downloads/km.bin"


class HubertCodes:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, kmeans_model_path=kmeans_model_path_global, max_chunk=1600000):
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").eval()#.cuda()
        self.max_chunk = max_chunk
        self.kmeans_model = joblib.load(open(kmeans_model_path, "rb"))
        self.kmeans_model.verbose = False

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        # assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None):
        x = self.read_audio(file_path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float()#.cuda()
            # x = torch.from_numpy(x).float().cuda()
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk = self.model(x_chunk)['last_hidden_state']
                feat.append(feat_chunk)
        # return torch.cat(feat, 1).squeeze(0)
        feats = torch.cat(feat, 1).squeeze(0)
        # K-means model
        pred = self.kmeans_model.predict(feats.cpu().numpy())
        return pred

    def get_feats_from_waveform(self, waveform):
        x = waveform
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            # x = x.float().cuda()
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                x_chunk = x_chunk.detach().cpu()
                feat_chunk = self.model(x_chunk)['last_hidden_state']
                feat.append(feat_chunk)
        # return torch.cat(feat, 1).squeeze(0)
        feats = torch.cat(feat, 1).squeeze(0).cpu().numpy()

        # K-means model
        pred = self.kmeans_model.predict(feats)#.cpu().numpy())
        return pred

    def get_feats_from_waveform_numpy(self, waveform):
        x = waveform
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk = self.model(x_chunk)['last_hidden_state']
                feat.append(feat_chunk)
        # return torch.cat(feat, 1).squeeze(0)
        feats = torch.cat(feat, 1).squeeze(0).cpu().numpy()

        # K-means model
        pred = self.kmeans_model.predict(feats)#.cpu().numpy())
        return pred


def get_feat_alon(file_path, kmeans_model_path):
    reader = HubertCodes(kmeans_model_path)

    pred = reader.get_feats(file_path)

    print(pred)


if __name__ == "__main__":
    # file_path = 'C:/git/speech-resynthesis-alon/data/VCTK-Corpus/wav16_silence_trimmed_padded/p263_021_mic2.flac'
    file_path = 'C:/git/speech-resynthesis-alon/data/VCTK-Corpus/wav16_silence_trimmed_padded/p374_313_mic2.flac'
    checkpoint_path = "C:/Users/User/Downloads/hubert_base_ls960.pt"
    layer: int = 6
    kmeans_model_path = "C:/Users/User/Downloads/km.bin"
    get_feat_alon(file_path,kmeans_model_path)
