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
from tqdm import tqdm

# import inference_alon

# import speaker_model
#
# speaker_m = speaker_model.SpeakerModel()
# speakers_data_path = r"src/speakers_data.csv"
# speakers_data = pd.read_csv(speakers_data_path).set_index('Id')
# speakers_data_dict = speakers_data.to_dict('index')
# speakers_list = []  # list(speakers_data_dict.keys())

import accent_model

accent_m = accent_model.AccentModel()
accent_mapping_dict = accent_m.get_accent_mapping()
inv_accent_mapping_dict = {v: k for k, v in accent_mapping_dict.items()}

# GENERATOR = inference_alon.init_generator('C:/git/speech-resynthesis-alon/tmp')
# speakers_list = GENERATOR.dataset.id_to_spkr


def build_data(rank, local_rank, a, h):
    device = torch.device('cuda:{:d}'.format(local_rank))
    print(f"Run on:{device}")

    training_filelist, validation_filelist = get_dataset_filelist(h)



    # trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
    #                        h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
    #                        device=device, f0=h.get('f0', None), multispkr=h.get('multispkr', None),
    #                        f0_stats=h.get('f0_stats', None),
    #                        f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
    #                        f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
    #                        vqvae=h.get('code_vq_params', False))
    # Change segment size to -1 from the
    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device, f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                           f0_stats=h.get('f0_stats', None),
                           f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                           f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                           vqvae=h.get('code_vq_params', False),
                           accent_embedding_by_accent=h.get('Accent_embedding_by_accent', None))



    # move to dataset init
    # num_of_splits_list = []
    # for index in tqdm(range(len(trainset))):
    #     num_of_splits = trainset._get_num_splits(index)
    #     num_of_splits_list.append(num_of_splits)
    #
    # np.save(r'vctk_balance/pkl/num_of_segments.npy', np.array(num_of_splits_list))
    #
    # training_files_list = []
    # training_codes_list = []
    # split_list = []
    # for training_files, training_codes, num_of_splits  in tqdm(zip(training_filelist[0], training_filelist[1], num_of_splits_list)):
    #     for split_index in range(num_of_splits + 1):
    #         training_files_list.append(training_files)
    #         training_codes_list.append(training_codes)
    #         split_list.append(split_index)
    #
    # list_of_tuples = list(zip(np.array(training_files_list), np.array(training_codes_list),np.array(split_list)))
    # input_train_df = pd.DataFrame(list_of_tuples, columns=['file_list', 'codes', 'split_index'])
    # # train_df.to_pickle('pkl/train_data_accent_distribution_all.pkl')


    # print(trainset[1])
    train_loader = DataLoader(trainset, num_workers=0, shuffle=False, sampler=None,
                              batch_size=1, pin_memory=True, drop_last=True)
    # batch_size=h.batch_size, pin_memory=True, drop_last=True)

    # validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
    #                        h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
    #                        fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
    #                        multispkr=h.get('multispkr', None),
    #                        f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
    #                        f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
    #                        f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False))
    # validset = CodeDataset(validation_filelist, -1, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
    #                        h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
    #                        fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
    #                        multispkr=h.get('multispkr', None),
    #                        f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
    #                        f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
    #                        f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),
    #                        accent_embedding_by_accent=h.get('Accent_embedding_by_accent', None))
    # validation_loader = DataLoader(validset, num_workers=0, shuffle=False, sampler=None,
    #                                batch_size=1, pin_memory=True, drop_last=True)
    # batch_size=h.batch_size, pin_memory=True, drop_last=True)

    # accent_list = []
    # accent_names = []
    # for i, batch in tqdm(enumerate(train_loader)):
    #     x, waveform, filename, _ = batch
    #
    #     accent_list.append(int(x['accent_id'].squeeze()))
    #     accent_names.append(accent_mapping_dict[int(x['accent_id'].squeeze())])
    #
    # list_of_tuples = list(zip(np.array(accent_names), np.array(accent_list)))
    # input_train_df = pd.DataFrame(list_of_tuples, columns=['accent_name', 'accent_id'])
    # input_train_df.to_pickle('vctk_balance/pkl/train_data_accent_distribution_balance_splits.pkl')


    x_list = []
    waveform_list = []
    filename_list = []
    mel_list = []
    for i, batch in tqdm(enumerate(train_loader)):
        x, waveform, filename, mel = batch
        x_list.append(x)
        waveform_list.append(waveform)
        filename_list.append(filename)
        mel_list.append(mel)

    list_of_tuples = list(zip(np.array(x_list), np.array(waveform_list),np.array(filename_list),np.array(mel_list)))
    input_train_df = pd.DataFrame(list_of_tuples, columns=['x', 'waveform', 'filename', 'mel'])
    input_train_df.to_pickle('vctk_balance/pkl/train_data_balance_splits.pkl')



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
