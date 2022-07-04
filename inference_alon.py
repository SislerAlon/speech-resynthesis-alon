# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import argparse
import glob
import json
import os
import pathlib
import random
import sys
import time
from multiprocessing import Manager, Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.io.wavfile import write

from dataset import CodeDataset, parse_manifest, mel_spectrogram, \
    MAX_WAV_VALUE
from utils import AttrDict
from models import CodeGenerator

h = None
device = None


def load_checkpoint(filepath):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cpu')
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def generate(h, generator, code):
    start = time.time()
    # Add here targer accent
    y_g_hat = generator(**code)
    if type(y_g_hat) is tuple:
        y_g_hat = y_g_hat[0]
    rtf = (time.time() - start) / (y_g_hat.shape[-1] / h.sampling_rate)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio, rtf


class GenerateWaveform:
    def __init__(self, a):
        self.device = torch.device('cuda:0')
        self.a = a
        self.output_dir = a.output_dir

        self.speaker_to_dataset_index = {}
        ## Speaker/accent data loading
        csv_path = pathlib.Path(r"src\speakers_data.csv")
        self.speaker_data = pd.read_csv(csv_path, index_col=0)
        # Example: speaker_data.loc['p225'].Accent
        ## Json loading
        if os.path.isdir(a.checkpoint_file):
            config_file = os.path.join(a.checkpoint_file, 'config.json')
        else:
            config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        # loading generator
        self.generator = CodeGenerator(self.h).to(self.device)


        if os.path.isdir(a.checkpoint_file):
            cp_g = scan_checkpoint(a.checkpoint_file, 'g_')
        else:
            cp_g = a.checkpoint_file
        state_dict_g = load_checkpoint(cp_g)
        self.generator.load_state_dict(state_dict_g['generator'])

        if a.code_file is not None:
            self.dataset = [x.strip().split('|') for x in open(a.code_file).readlines()]

            def parse_code(c):
                c = [int(v) for v in c.split(" ")]
                return [torch.LongTensor(c).numpy()]

            self.dataset = [(parse_code(x[1]), None, x[0], None) for x in self.dataset]
        else:
            file_list = parse_manifest(self.a.input_code_file)
            self.dataset = CodeDataset(file_list, -1, self.h.code_hop_size, self.h.n_fft, self.h.num_mels, self.h.hop_size, self.h.win_size,
                                       self.h.sampling_rate, self.h.fmin, self.h.fmax, n_cache_reuse=0,
                                       fmax_loss=self.h.fmax_for_loss, device=device,
                                       f0=self.h.get('f0', None), multispkr=self.h.get('multispkr', None),
                                       f0_stats=self.h.get('f0_stats', None), f0_normalize=self.h.get('f0_normalize', False),
                                       f0_feats=self.h.get('f0_feats', False), f0_median=self.h.get('f0_median', False),
                                       f0_interp=self.h.get('f0_interp', False), vqvae=self.h.get('code_vq_params', False),
                                       pad=a.pad, accent_embedding_by_accent=self.h.get('Accent_embedding_by_accent', None))

        if self.a.unseen_f0:
            self.dataset.f0_stats = torch.load(self.a.unseen_f0)

        os.makedirs(self.a.output_dir, exist_ok=True)


        # if a.f0_stats and h.get('f0', None) is not None:
        #     f0_stats = torch.load(a.f0_stats)

        self.generator.eval()
        self.generator.remove_weight_norm()

        # fix seed
        seed = 52
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def get_index_by_speaker(self):
        if self.speaker_to_dataset_index == {}:
            for index, audio_file in enumerate(self.dataset.audio_files):
                speaker = audio_file.stem.split('_')[0]
                utterance_index = audio_file.stem.split('_')[1]
                text_file_path = Path(audio_file.parts[0]) / audio_file.parts[1] / 'txt'/ speaker / f'{speaker}_{utterance_index}.txt'
                with open(text_file_path, 'r') as f:
                    text_content = f.readline().strip()
                data = {'index': index, 'transcription': text_content}
                if speaker in self.speaker_to_dataset_index:
                    self.speaker_to_dataset_index[speaker].append(data)
                else:
                    self.speaker_to_dataset_index[speaker] = [data]
        return self.speaker_to_dataset_index

    @torch.no_grad()
    def __call__(self, item_index, target_speaker='p225', target_accent='p226'):#target_accent='American'):
        sound_file_output_path = {}
        code, gt_audio, filename, _ = self.dataset[item_index]
        code = {k: torch.from_numpy(v).to(self.device).unsqueeze(0) for k, v in code.items()}
        original_speaker = self.dataset.id_to_spkr[int(code['spkr'][0][0])]
        original_accent = self.generator.speakers_data_dict[original_speaker]['Accent']
        # original_accent = self.speaker_data.loc[original_speaker].Accent
        # # if a.parts:
        # if False:
        #     parts = Path(filename).parts
        #     fname_out_name = '_'.join(parts[-3:])[:-4]
        # else:
        fname_out_name = Path(filename).stem

        # if self.a.dur_prediction:
        #     code['code'] = torch.unique_consecutive(code['code']).unsqueeze(0)
        #     code['dur_prediction'] = True


        if self.h.get('f0_vq_params', None) or self.h.get('f0_quantizer', None):
            # משהו פה
            to_remove = gt_audio.shape[-1] % (16 * 80)
            assert to_remove % self.h['code_hop_size'] == 0

            if to_remove != 0:
                to_remove_code = to_remove // self.h['code_hop_size']
                to_remove_f0 = to_remove // 80

                gt_audio = gt_audio[:-to_remove]
                code['code'] = code['code'][..., :-to_remove_code]
                code['f0'] = code['f0'][..., :-to_remove_f0]

        new_code = dict(code)
        if 'f0' in new_code:
            del new_code['f0']
            new_code['f0'] = code['f0']

        # if self.a.dur_prediction:
        #     new_code['code'] = torch.unique_consecutive(new_code['code']).unsqueeze(0)
        #     new_code['dur_prediction'] = True

        # re-synth
        re_synth_orig = True
        if re_synth_orig:
            audio, rtf = generate(self.h, self.generator, new_code)
            # output_file = os.path.join(self.output_dir, fname_out_name + '_gen_orig.wav')
            output_file = os.path.join(self.output_dir, fname_out_name + f'_speaker_{original_speaker}_accent_{original_accent}_gen-orig.wav')
            sound_file_output_path['gt_gen'] = output_file

            audio = librosa.util.normalize(audio.astype(np.float32))
            write(output_file, self.h.sampling_rate, audio)

        # if self.h.get('multispkr', None) and self.a.vc:
        if original_speaker != target_speaker or original_accent != target_accent:
            local_spkrs = [self.dataset.spkr_to_id[target_speaker]]
            for spkr_i, k in enumerate(local_spkrs):
                code['accent_id'] = code['spkr'].clone()
                # new_code['accent'] = new_code['spkr'].clone()
                code['spkr'].fill_(k)
                # new_code['spkr'].fill_(k)
                # code['accent'].fill_(self.dataset.spkr_to_id[target_accent])
                code['accent_id'].fill_(self.dataset.accent_to_id_mapping[target_accent])
                # new_code['accent'].fill_(self.dataset.spkr_to_id[target_accent])
                # Not use
                if self.a.f0_stats and self.h.get('f0', None) is not None and not self.h.get('f0_normalize', False):
                    spkr = k
                    # f0 = code['f0'].clone()
                    f0 = new_code['f0'].clone()

                    ii = (f0 != 0)
                    mean_, std_ = f0[ii].mean(), f0[ii].std()
                    if spkr not in f0_stats:
                        new_mean_, new_std_ = f0_stats['f0_mean'], f0_stats['f0_std']
                    else:
                        new_mean_, new_std_ = f0_stats[spkr]['f0_mean'], f0_stats[spkr]['f0_std']

                    f0[ii] -= mean_
                    f0[ii] /= std_
                    f0[ii] *= new_std_
                    f0[ii] += new_mean_
                    # code['f0'] = f0
                    new_code['f0'] = f0

                # Not use
                if self.h.get('f0_feats', False):
                    f0_stats_ = torch.load(self.h["f0_stats"])
                    if k not in f0_stats_:
                        mean = f0_stats_['f0_mean']
                        std = f0_stats_['f0_std']
                    else:
                        mean = f0_stats_[k]['f0_mean']
                        std = f0_stats_[k]['f0_std']
                    # code['f0_stats'] = torch.FloatTensor([mean, std]).view(1, -1).to(device)
                    new_code['f0_stats'] = torch.FloatTensor([mean, std]).view(1, -1).to(device)

                audio, rtf = generate(self.h, self.generator, code)
                # audio, rtf = generate(self.h, self.generator, new_code)

                output_file = os.path.join(self.output_dir, fname_out_name + f'_speaker_{target_speaker}_accent_{target_accent}_gen.wav')
                sound_file_output_path['new'] = output_file
                audio = librosa.util.normalize(audio.astype(np.float32))
                write(output_file, self.h.sampling_rate, audio)

        if gt_audio is not None:
            # output_file = os.path.join(self.output_dir, fname_out_name + '_gt.wav')
            output_file = os.path.join(self.output_dir, fname_out_name + f'_accent_{original_accent}_gt.wav')
            gt_audio = librosa.util.normalize(gt_audio.squeeze().numpy().astype(np.float32))
            sound_file_output_path['gt'] = output_file
            write(output_file, self.h.sampling_rate, gt_audio)

        return sound_file_output_path



def init_generator(main_output_path='D:/Thesis/generated_results'):
    print('Initializing Inference Process..')
    # main_output_path = 'D:/Thesis/generated_results'
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', default=None)
    parser.add_argument('--input_code_file', default='./datasets/VCTK/hubert100/test.txt')
    parser.add_argument('--output_dir', default=f'{main_output_path}/generated_files_accent_change_double_zero_accent_duration')
    # parser.add_argument('--checkpoint_file', default='checkpoints/VCTK_vqvae_accent_speaker')  # , required=True)
    # parser.add_argument('--checkpoint_file', default='checkpoints/VCTK_vqvae_accent_speaker_duration')  # , required=True)
    parser.add_argument('--checkpoint_file', default='checkpoints/VCTK_vqvae_accent_speaker_duration_by_accent')  # , required=True)
    # parser.add_argument('--checkpoint_file', default='checkpoints/VCTK_vqvae', required=True)
    parser.add_argument('--f0-stats', type=Path)
    parser.add_argument('--vc', action='store_true')
    parser.add_argument('--random-speakers', action='store_true')
    parser.add_argument('--pad', default=None, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--parts', action='store_true')
    parser.add_argument('--unseen-f0', type=Path)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('--dur-prediction', default=True)# ,action='store_true')
    a = parser.parse_args()

    generator = GenerateWaveform(a)
    return generator

def generate_from_speakers(generator, base_speaker,target_speaker,target_accent,base_speaker_index=0):
    speaker_to_dataset_index = generator.get_index_by_speaker()


    item_index = speaker_to_dataset_index[base_speaker][base_speaker_index]['index']

    # print(speaker_to_dataset_index[base_speaker])
    sound_file_output_path = generator(item_index=item_index,
                                      target_speaker=target_speaker,
                                      target_accent=target_accent)
    return sound_file_output_path


def main():
    generator = init_generator()
    base_speaker = 'p225'# 'p226'
    target_speaker = 'p225'# 'p248'
    target_accent = 'American'# 'p249'
    sound_file_output_path = generate_from_speakers(generator=generator,
                           base_speaker=base_speaker,
                           target_speaker=target_speaker,
                           target_accent=target_accent)
    print(sound_file_output_path)


if __name__ == '__main__':
    main()
