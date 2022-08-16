import torchaudio
import torch
import UTMOS.lightning_module as lightning_module
import pathlib


class MosModel:
    description = """
    MOS prediction demo using UTMOS-strong w/o phoneme encoder model, which is trained on the main track dataset.
    This demo only accepts .wav format. Best at 16 kHz sampling rate.
    Paper is available [here](https://arxiv.org/abs/2204.02152)
    """
    def __init__(self):
        print("Load Mos model")
        model_cp_path = str(pathlib.Path(__file__).parent.resolve()/"epoch=3-step=7459.ckpt")
        self.model = lightning_module.BaselineLightningModule.load_from_checkpoint(model_cp_path).eval()


    def __call__(self, waveform):#audio_path):
        # wav, sr = torchaudio.load(audio_path)
        osr = 16_000
        out_wavs = waveform
        batch = {
            'wav': out_wavs,
            'domains': torch.tensor([0]),
            'judge_id': torch.tensor([288])
        }
        with torch.no_grad():
            output = self.model(batch)
        return output.mean(dim=1).squeeze().detach().numpy() * 2 + 3


if __name__ == "__main__":
    mm = MosModel()
    waveform_path = r"C:\git\speech-resynthesis-alon\tmp\balance_dataset\p225_084_mic2_accent_English_gt.wav"
    wav, sr = torchaudio.load(waveform_path)
    print(mm(wav))
    print("Bammmm")