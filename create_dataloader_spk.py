import torch
from speechbrain.pretrained import EncoderClassifier
from utils import AttrDict
import json
from dataset import CodeDataset, get_dataset_filelist
import argparse
import os
from utils import AttrDict
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import soundfile as sf

from sklearn import preprocessing





def build_data(rank, local_rank, a, h):
    device = torch.device('cuda:{:d}'.format(local_rank))
    print(f"Run on:{device}")

    training_filelist, validation_filelist = get_dataset_filelist(h)


    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device, f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                           f0_stats=h.get('f0_stats', None),
                           f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                           f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                           vqvae=h.get('code_vq_params', False))

    train_loader = DataLoader(trainset, num_workers=0, shuffle=False, sampler=None,
                              batch_size=1, pin_memory=True, drop_last=True)
                              # batch_size=h.batch_size, pin_memory=True, drop_last=True)

    validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
                           fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
                           multispkr=h.get('multispkr', None),
                           f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                           f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                           f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False))
    validation_loader = DataLoader(validset, num_workers=0, shuffle=False, sampler=None,
                                   batch_size=1, pin_memory=True, drop_last=True)
                                   # batch_size=h.batch_size, pin_memory=True, drop_last=True)

    base_x_vectors = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                         savedir="pretrained_models/spkrec-xvect-voxceleb")


    feat_list = []#None
    spk_list = []#None

    for i, batch in enumerate(train_loader):
        # print(f"Train Index: {i}", end='\r')
        print(f"Train Index: {i}")
        # x, waveform, _, _ = batch
        x, waveform, filename, _ = batch
        # waveform, sampling_rate = sf.read(filename[0])
        inputs = base_x_vectors.encode_batch(torch.Tensor(waveform)).squeeze()

        # inputs = base_x_vectors.encode_batch(waveform).squeeze()
        feat_list.append(inputs)
        # if feat_list == None:
        #     feat_list = inputs
        # else:
        #     feat_list = torch.cat((feat_list, inputs))

        spk_list.append(int(x['spkr'].squeeze()))

        # if spk_list == None:
        #     spk_list = x['spkr'].squeeze()
        # else:
        #     spk_list = torch.cat((spk_list, x['spkr'].squeeze()))

    # fitter = preprocessing.StandardScaler().fit(torch.stack(feat_list).numpy())
    # feat_list_norm = fitter.transform(torch.stack(feat_list).numpy())


    # list_of_tuples = list(zip(feat_list_norm, np.array(spk_list)))
    # list_of_tuples = list(zip(feat_list.numpy(), spk_list.numpy()))
    # list_of_tuples = list(zip(np.array(feat_list.numpy()), np.array(spk_list)))
    list_of_tuples = list(zip(torch.stack(feat_list).numpy(), np.array(spk_list)))
    train_df = pd.DataFrame(list_of_tuples, columns=['Features', 'speaker_id'])
    # train_df.to_pickle('train_data_norm.pkl')
    train_df.to_pickle('train_data_orig.pkl')


    feat_list = []#None
    spk_list = []#None

    for i, batch in enumerate(validation_loader):
        # print(f"Validation Index: {i}", end='\r')
        print(f"Validation Index: {i}")
        x, waveform, filename, _ = batch
        # waveform, sampling_rate = sf.read(filename[0])

        inputs = base_x_vectors.encode_batch(torch.Tensor(waveform)).squeeze()

        feat_list.append(inputs)
        # if feat_list == None:
        #     feat_list = inputs
        # else:
        #     feat_list = torch.cat((feat_list, inputs))

        spk_list.append(int(x['spkr'].squeeze()))

        # if spk_list == None:
        #     spk_list = x['spkr'].squeeze()
        # else:
        #     spk_list = torch.cat((spk_list, x['spkr'].squeeze()))

    # same fitter for train and validation
    # fitter = preprocessing.StandardScaler().fit(torch.stack(feat_list).numpy())
    # feat_list_norm = fitter.transform(torch.stack(feat_list).numpy())

    # list_of_tuples = list(zip(feat_list_norm, np.array(spk_list)))
    # list_of_tuples = list(zip(feat_list.numpy(), spk_list.numpy()))
    list_of_tuples = list(zip(torch.stack(feat_list).numpy(), np.array(spk_list)))

    val_df = pd.DataFrame(list_of_tuples, columns=['Features', 'speaker_id'])
    # val_df.to_pickle('val_data_norm.pkl')
    val_df.to_pickle('val_data_orig.pkl')

























def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='checkpoints/VCTK_vqvae_accent_speaker_speaker')
    parser.add_argument('--config', default='configs/VCTK/hubert100_lut.json')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--training_steps', default=400000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed-world-size', type=int)
    parser.add_argument('--distributed-port', type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    rank = 0
    local_rank = 0

    build_data(rank, local_rank, a, h)

if __name__ == '__main__':
    main()