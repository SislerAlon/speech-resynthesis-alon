# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import CodeDataset, mel_spectrogram, get_dataset_filelist
from models import CodeGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict
import Hubert_Codes


# import speaker_model
# import numpy as np
# import librosa
# import soundfile as sf

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


import evaluator
import numpy as np

def generate_exclude_random_tensor(original_tensor, device, batch_size=16):
    all_options_array = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]] * batch_size)
    '''
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
    '''
    same_indexes = np.where(original_tensor.cpu().numpy() == all_options_array)
    indexes_to_remove_from_choise = [[row * 11 + col] for row, col in zip(same_indexes[0], same_indexes[1])]
    matrix_without_the_original_ids = np.delete(all_options_array, indexes_to_remove_from_choise).reshape(batch_size, 10)
    samples_values = matrix_without_the_original_ids[np.arange(batch_size), np.random.randint(0, 10, (batch_size,))]
    samples_vector = torch.from_numpy(samples_values.reshape(batch_size, 1)).to(device)
    return samples_vector.to(torch.int64)









def train(rank, local_rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.num_gpus,
        )
    torch.autograd.set_detect_anomaly(True)
    hub_codes = Hubert_Codes.HubertCodes()

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))

    generator = CodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

        # import gc
        # del state_dict_g
        # del state_dict_do
        # state_dict_do = None
        # gc.collect()

    # if h.num_gpus > 1:
    #     generator = DistributedDataParallel(
    #         generator,
    #         device_ids=[local_rank],
    #         find_unused_parameters=('f0_quantizer' in h),
    #     ).to(device)
    #     mpd = DistributedDataParallel(mpd, device_ids=[local_rank]).to(device)
    #     msd = DistributedDataParallel(msd, device_ids=[local_rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    ##only if load pretrained
    import gc
    del state_dict_g
    del state_dict_do
    gc.collect()
    ##

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device, f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                           f0_stats=h.get('f0_stats', None),
                           f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                           f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                           vqvae=h.get('code_vq_params', False),accent_embedding_by_accent=h.get('Accent_embedding_by_accent', None))

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    num_workers = 6#2#0#5#4
    train_loader = DataLoader(trainset, num_workers=num_workers, shuffle=True, sampler=train_sampler,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)
    # train_loader = DataLoader(trainset, num_workers=num_workers, shuffle=False, sampler=train_sampler,
    #                           batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validset = CodeDataset(validation_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                               h.win_size, h.sampling_rate, h.fmin, h.fmax, False, n_cache_reuse=0,
                               fmax_loss=h.fmax_for_loss, device=device, f0=h.get('f0', None),
                               multispkr=h.get('multispkr', None),
                               f0_stats=h.get('f0_stats', None), f0_normalize=h.get('f0_normalize', False),
                               f0_feats=h.get('f0_feats', False), f0_median=h.get('f0_median', False),
                               f0_interp=h.get('f0_interp', False), vqvae=h.get('code_vq_params', False),accent_embedding_by_accent=h.get('Accent_embedding_by_accent', None),
                               validation_data=True)
        validation_loader = DataLoader(validset, num_workers=num_workers, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    speakers_list = trainset.id_to_spkr
    evaluator_models = evaluator.Evaluator(speakers_list=speakers_list)
    add_speaker_loss = True
    add_accent_loss = True
    speaker_loss = torch.nn.CrossEntropyLoss()
    accent_loss = torch.nn.CrossEntropyLoss()
    speaker_const = 10.
    accent_const = 10.


    accent_to_id_mapping = trainset.accent_to_id_mapping
    add_multi_accent_loss = True

    multi_accent_loss = torch.nn.CrossEntropyLoss()
    multi_accent_const = 10. #accent_const / 10.
    multi_speaker_loss = torch.nn.CrossEntropyLoss()
    multi_speaker_const = 10. #accent_const / 10.

    cycle_loss = True
    speaker_loss_cycle = torch.nn.CrossEntropyLoss()
    accent_loss_cycle = torch.nn.CrossEntropyLoss()
    speaker_const_cycle = 10.
    accent_const_cycle = 10.


    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            # x, y, _, y_mel = batch
            x, y, file_name_list, y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            y = y.unsqueeze(1)
            x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}

            # y_g_hat = generator(**x)
            y_g_hat, dur_losses = generator(**x)

            if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                y_g_hat, commit_losses, metrics = y_g_hat

            assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"
            if h.get('f0_vq_params', None):
                f0_commit_loss = commit_losses[1][0]
                f0_metrics = metrics[1][0]
            if h.get('code_vq_params', None):
                code_commit_loss = commit_losses[0][0]
                code_metrics = metrics[0][0]

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            optim_d.zero_grad(set_to_none=True)

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad(set_to_none=True)

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            if h.get('f0_vq_params', None):
                loss_gen_all += f0_commit_loss * h.get('lambda_commit', None)
            if h.get('code_vq_params', None):
                loss_gen_all += code_commit_loss * h.get('lambda_commit_code', None)
            if h.get('dur_prediction_weight', None):
                loss_gen_all += dur_losses * h.get('dur_prediction_weight', None)


            if add_speaker_loss:
                classified_speaker = evaluator_models.evaluate_speaker(y_g_hat.cpu().detach(), final_prediction=False)
                speaker_loss_val = speaker_const * speaker_loss(torch.Tensor(classified_speaker).cpu(), x['spkr'].cpu().detach().squeeze())
                loss_gen_all += speaker_loss_val
            if add_accent_loss:
                classified_accent = evaluator_models.evaluate_accent(y_g_hat.cpu().detach(), final_prediction=False)
                accent_loss_val = accent_const * accent_loss(torch.Tensor(classified_accent).cpu(), x['accent_id'].cpu().detach().squeeze())
                loss_gen_all += accent_loss_val

            if add_multi_accent_loss:
                original_accent_id_list = x['accent_id'].detach().clone()
                # x_modify = x.copy()
                x_modify = {}
                x_modify['code'] = x['code'].detach().clone()
                x_modify['accent_id'] = x['accent_id'].detach().clone()
                x_modify['f0'] = x['f0'].detach().clone()
                x_modify['spkr'] = x['spkr'].detach().clone()






                new_accent_to_generate = generate_exclude_random_tensor(original_accent_id_list, device, batch_size=16)

                # Modify X
                x_modify['accent_id'] = new_accent_to_generate

                # generate accents
                y_g_hat_multi, dur_losses = generator(**x_modify)

                # classify accent
                classified_accent_multi = evaluator_models.evaluate_accent(y_g_hat_multi.cpu().detach(), final_prediction=False)

                multi_accent_loss_val = multi_accent_const * multi_accent_loss(torch.Tensor(classified_accent_multi).cpu(),
                                                                               x_modify[
                                                                                   'accent_id'].cpu().detach().squeeze())
                loss_gen_all += multi_accent_loss_val

                # classify speaker
                classified_speaker_multi = evaluator_models.evaluate_speaker(y_g_hat_multi.cpu().detach(), final_prediction=False)

                multi_speaker_loss_val = multi_speaker_const * multi_speaker_loss(torch.Tensor(classified_speaker_multi).cpu(),
                                                                               x_modify[
                                                                                   'spkr'].cpu().detach().squeeze())

                loss_gen_all += multi_speaker_loss_val

            if cycle_loss and add_multi_accent_loss:

                # x_cycle = x_modify.copy()
                x_cycle = {}
                x_cycle['code'] = x_modify['code'].detach().clone()
                x_cycle['accent_id'] = x_modify['accent_id'].detach().clone()
                x_cycle['f0'] = x_modify['f0'].detach().clone()
                x_cycle['spkr'] = x_modify['spkr'].detach().clone()


                y_g_hat_multi_copy = y_g_hat_multi.detach().clone()
                speaker_orig_list = x['spkr'].detach().clone()
                for index, (waveform_idx, spkr_id) in enumerate(zip(y_g_hat_multi_copy, speaker_orig_list)):
                    cycle_feats = trainset.get_item_from_waveform_for_cycle_loss(waveform_idx, int(spkr_id[0]), hub_codes)
                    # cycle_feats = trainset.get_item_from_waveform_for_cycle_loss(waveform_idx, int(spkr_id[0]))
                    x_cycle['f0'][index] = torch.from_numpy(cycle_feats['f0']).to(device)
                    x_cycle['code'][index] = torch.from_numpy(cycle_feats['code']).to(device) ###########


                # accent, get the original accent
                x_cycle['accent_id'] = original_accent_id_list.detach().clone()
                # get it from the original
                x_cycle['spkr'] = x['spkr'].detach().clone()

                y_g_hat_cycle, dur_losses_cycle = generator(**x_cycle)

                if add_speaker_loss:
                    classified_speaker_cycle = evaluator_models.evaluate_speaker(y_g_hat_cycle.cpu().detach(),
                                                                           final_prediction=False)
                    speaker_loss_val_cycle = speaker_const_cycle * speaker_loss_cycle(torch.Tensor(classified_speaker_cycle).cpu(),
                                                                    x_cycle['spkr'].cpu().detach().squeeze())
                    loss_gen_all += speaker_loss_val_cycle
                if add_accent_loss:
                    classified_accent_cycle = evaluator_models.evaluate_accent(y_g_hat_cycle.cpu().detach(), final_prediction=False)
                    accent_loss_val_cycle = accent_const_cycle * accent_loss_cycle(torch.Tensor(classified_accent_cycle).cpu(),
                                                                 x_cycle['accent_id'].cpu().detach().squeeze())
                    loss_gen_all += accent_loss_val_cycle

                y_g_hat_cycle_mel = mel_spectrogram(y_g_hat_cycle.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                              h.win_size, h.fmin, h.fmax_for_loss)

                # L1 Mel-Spectrogram Loss
                loss_mel_cycle = F.l1_loss(y_mel, y_g_hat_cycle_mel) * 45
                loss_gen_all += loss_mel_cycle

            else:
                speaker_loss_val_cycle = -1
                accent_loss_val_cycle = -1
                loss_mel_cycle = -1


            loss_gen_all.backward()
            optim_g.step()
            # print("Bammmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                        if add_speaker_loss:
                            classified_speaker = evaluator_models.evaluate_speaker(y_g_hat.cpu().detach(),
                                                                                   final_prediction=False)
                            speaker_loss_val = speaker_loss(torch.Tensor(classified_speaker).cpu(),
                                                                            x['spkr'].cpu().detach().squeeze())
                        if add_accent_loss:
                            classified_accent = evaluator_models.evaluate_accent(y_g_hat.cpu().detach(),
                                                                                 final_prediction=False)
                            accent_loss_val = accent_loss(torch.Tensor(classified_accent).cpu(),
                                                                         x['accent_id'].cpu().detach().squeeze())

                        if add_multi_accent_loss:
                            # classify accent
                            classified_accent_multi = evaluator_models.evaluate_accent(y_g_hat_multi.cpu().detach(),
                                                                                       final_prediction=False)

                            multi_accent_val = multi_accent_loss(
                                torch.Tensor(classified_accent_multi).cpu(),
                                x_modify[
                                    'accent_id'].cpu().detach().squeeze())

                            # classify speaker
                            classified_speaker_multi = evaluator_models.evaluate_speaker(y_g_hat_multi.cpu().detach(),
                                                                                         final_prediction=False)

                            multi_speaker_val = multi_speaker_loss(
                                torch.Tensor(classified_speaker_multi).cpu(),
                                x_modify[
                                    'spkr'].cpu().detach().squeeze())

                        if add_multi_accent_loss and cycle_loss:
                            loss_mel_cycle = F.l1_loss(y_mel, y_g_hat_cycle_mel).item()
                            classified_speaker_cycle = evaluator_models.evaluate_speaker(y_g_hat_cycle.cpu().detach(),
                                                                                         final_prediction=False)
                            speaker_loss_val_cycle = speaker_loss_cycle(
                                torch.Tensor(classified_speaker_cycle).cpu(),
                                x_cycle['spkr'].cpu().detach().squeeze())


                            classified_accent_cycle = evaluator_models.evaluate_accent(y_g_hat_cycle.cpu().detach(),
                                                                                       final_prediction=False)
                            accent_loss_val_cycle = accent_loss_cycle(
                                torch.Tensor(classified_accent_cycle).cpu(),
                                x_cycle['accent_id'].cpu().detach().squeeze())



                    print(f'Steps : {steps:d}, '\
                          f'Gen Loss Total : {loss_gen_all:4.3f}, ' \
                          f'Mel-Spec. Error : {mel_error:4.3f}, '\
                          f'Speaker Error : {speaker_loss_val:4.3f}, '\
                          f'Accent Error : {accent_loss_val:4.3f}, '\
                          f's/b : {time.time() - start_b:4.3f}, '\
                          f'Speaker Error Multi: {multi_speaker_val:4.3f}, '\
                          f'Accent Error Multi: {multi_accent_val:4.3f}, ' \
                          f'Speaker Error Cycle: {speaker_loss_val_cycle:4.3f}, ' \
                          f'Accent Error Cycle: {accent_loss_val_cycle:4.3f}, '\
                          f'Mel-Spec. Error Cycle: {loss_mel_cycle:4.3f}')
                    # print(
                    #     'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f},Speaker Error : {:4.3f},Accent Error : {:4.3f}, s/b : {:4.3f}'.format(steps,
                    #                                                                                               loss_gen_all,
                    #                                                                                               mel_error,
                    #                                                                                               speaker_loss_val,
                    #                                                                                               accent_loss_val,
                    #                                                                                               time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                      'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                                      'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                      'steps': steps, 'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    if add_speaker_loss:
                        sw.add_scalar("training/speaker_loss", speaker_loss_val, steps)
                    if add_accent_loss:
                        sw.add_scalar("training/accent_loss", accent_loss_val, steps)
                    if add_multi_accent_loss:
                        sw.add_scalar("training/Multi_accent_classification_error", multi_accent_val, steps)
                        sw.add_scalar("training/Multi_speaker_classification_error", multi_speaker_val, steps)
                        if cycle_loss:
                            sw.add_scalar("training/Cycle_accent_classification_error", accent_loss_val_cycle, steps)
                            sw.add_scalar("training/Cycle_speaker_classification_error", speaker_loss_val_cycle, steps)
                            sw.add_scalar("training/Cycle_mel_spec_error", loss_mel_cycle, steps)


                    if h.get('f0_vq_params', None):
                        sw.add_scalar("training/commit_error", f0_commit_loss, steps)
                        sw.add_scalar("training/used_curr", f0_metrics['used_curr'].item(), steps)
                        sw.add_scalar("training/entropy", f0_metrics['entropy'].item(), steps)
                        sw.add_scalar("training/usage", f0_metrics['usage'].item(), steps)
                    if h.get('code_vq_params', None):
                        sw.add_scalar("training/code_commit_error", code_commit_loss, steps)
                        sw.add_scalar("training/code_used_curr", code_metrics['used_curr'].item(), steps)
                        sw.add_scalar("training/code_entropy", code_metrics['entropy'].item(), steps)
                        sw.add_scalar("training/code_usage", code_metrics['usage'].item(), steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    speaker_loss_val_tot = 0
                    accent_loss_val_tot = 0

                    multi_accent_loss_val_tot = 0
                    multi_speaker_loss_val_tot = 0
                    cycle_accent_loss_val_tot = 0
                    cycle_speaker_loss_val_tot = 0
                    cycle_mel_val_err_tot = 0

                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            x = {k: v.to(device, non_blocking=False) for k, v in x.items()}

                            # y_g_hat = generator(**x)
                            y_g_hat, dur_losses = generator(**x)

                            if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                                y_g_hat, commit_losses, _ = y_g_hat

                            if h.get('f0_vq_params', None):
                                f0_commit_loss = commit_losses[1][0]
                                val_err_tot += f0_commit_loss * h.get('lambda_commit', None)

                            if h.get('code_vq_params', None):
                                code_commit_loss = commit_losses[0][0]
                                val_err_tot += code_commit_loss * h.get('lambda_commit_code', None)
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if add_speaker_loss:
                                classified_speaker = evaluator_models.evaluate_speaker(y_g_hat.cpu().detach(),
                                                                                       final_prediction=False)
                                speaker_loss_val_tot += speaker_const * speaker_loss(torch.Tensor(classified_speaker).cpu(),
                                                                                x['spkr'].cpu().detach().squeeze())
                                # val_err_tot += speaker_loss_val
                            if add_accent_loss:
                                classified_accent = evaluator_models.evaluate_accent(y_g_hat.cpu().detach(),
                                                                                     final_prediction=False)
                                accent_loss_val_tot += accent_const * accent_loss(torch.Tensor(classified_accent).cpu(),
                                                                             x['accent_id'].cpu().detach().squeeze())
                                # val_err_tot += accent_loss_val

                            if add_multi_accent_loss:
                                original_accent_id_list = x['accent_id']
                                # x_modify = x.copy()
                                x_modify = {}
                                x_modify['code'] = x['code'].detach().clone()
                                x_modify['accent_id'] = x['accent_id'].detach().clone()
                                x_modify['f0'] = x['f0'].detach().clone()
                                x_modify['spkr'] = x['spkr'].detach().clone()

                                new_accent_to_generate = generate_exclude_random_tensor(original_accent_id_list, device,
                                                                                        batch_size=16)

                                # Modify X
                                x_modify['accent_id'] = new_accent_to_generate

                                # generate accents
                                y_g_hat_multi, dur_losses = generator(**x_modify)

                                # classify accent
                                classified_accent_multi = evaluator_models.evaluate_accent(y_g_hat_multi.cpu().detach(),
                                                                                           final_prediction=False)

                                multi_accent_loss_val = multi_accent_const * multi_accent_loss(
                                    torch.Tensor(classified_accent_multi).cpu(),
                                    x_modify[
                                        'accent_id'].cpu().detach().squeeze())
                                multi_accent_loss_val_tot += multi_accent_loss_val

                                # classify speaker
                                classified_speaker_multi = evaluator_models.evaluate_speaker(
                                    y_g_hat_multi.cpu().detach(), final_prediction=False)

                                multi_speaker_loss_val = multi_speaker_const * multi_speaker_loss(
                                    torch.Tensor(classified_speaker_multi).cpu(),
                                    x_modify[
                                        'spkr'].cpu().detach().squeeze())

                                multi_speaker_loss_val_tot += multi_speaker_loss_val

                            if cycle_loss and add_multi_accent_loss:
                                # x_cycle = x_modify.copy()
                                x_cycle = {}
                                x_cycle['code'] = x_modify['code'].detach().clone()
                                x_cycle['accent_id'] = x_modify['accent_id'].detach().clone()
                                x_cycle['f0'] = x_modify['f0'].detach().clone()
                                x_cycle['spkr'] = x_modify['spkr'].detach().clone()



                                for index, (waveform_idx, spkr_id) in enumerate(zip(y_g_hat_multi, x['spkr'])):
                                    feats = trainset.get_item_from_waveform_for_cycle_loss(waveform_idx,
                                                                                           int(spkr_id[0]), hub_codes)
                                    # feats = trainset.get_item_from_waveform_for_cycle_loss(waveform_idx,
                                    #                                                        int(spkr_id[0]))
                                    x_cycle['f0'][index] = torch.from_numpy(feats['f0']).to(device)
                                    x_cycle['code'][index] = torch.from_numpy(feats['code']).to(device)

                                # accent, get the original accent
                                x_cycle['accent_id'] = original_accent_id_list
                                # get it from the original
                                x_cycle['spkr'] = x['spkr']

                                y_g_hat_cycle, dur_losses_cycle = generator(**x_cycle)

                                if add_speaker_loss:
                                    classified_speaker_cycle = evaluator_models.evaluate_speaker(
                                        y_g_hat_cycle.cpu().detach(),
                                        final_prediction=False)
                                    speaker_loss_val_cycle = speaker_const_cycle * speaker_loss_cycle(
                                        torch.Tensor(classified_speaker_cycle).cpu(),
                                        x_cycle['spkr'].cpu().detach().squeeze())
                                    cycle_speaker_loss_val_tot += speaker_loss_val_cycle
                                if add_accent_loss:
                                    classified_accent_cycle = evaluator_models.evaluate_accent(
                                        y_g_hat_cycle.cpu().detach(), final_prediction=False)
                                    accent_loss_val_cycle = accent_const_cycle * accent_loss_cycle(
                                        torch.Tensor(classified_accent_cycle).cpu(),
                                        x_cycle['accent_id'].cpu().detach().squeeze())
                                    cycle_accent_loss_val_tot += accent_loss_val_cycle

                                y_g_hat_cycle_mel = mel_spectrogram(y_g_hat_cycle.squeeze(1), h.n_fft, h.num_mels,
                                                                    h.sampling_rate, h.hop_size,
                                                                    h.win_size, h.fmin, h.fmax_for_loss)

                                # L1 Mel-Spectrogram Loss
                                loss_mel_cycle = F.l1_loss(y_mel, y_g_hat_cycle_mel) * 45
                                cycle_mel_val_err_tot += loss_mel_cycle




                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(y_mel[0].cpu()), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat[:1].squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec[:1].squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                        if add_speaker_loss:
                            speaker_loss_val_err = speaker_loss_val_tot / (j + 1)
                            sw.add_scalar("validation/speaker_classification_error", speaker_loss_val_err, steps)

                        if add_accent_loss:
                            accent_loss_val_err = accent_loss_val_tot / (j + 1)
                            sw.add_scalar("validation/accent_classification_error", accent_loss_val_err, steps)

                        if add_multi_accent_loss:
                            multi_accent_loss_val_err = multi_accent_loss_val_tot / (j + 1)
                            sw.add_scalar("validation/Multi_accent_classification_error", multi_accent_loss_val_err, steps)

                            multi_speaker_loss_val_err = multi_speaker_loss_val_tot / (j + 1)
                            sw.add_scalar("validation/Multi_speaker_classification_error", multi_speaker_loss_val_err, steps)
                            if cycle_loss:
                                cycle_accent_loss_val_err = cycle_accent_loss_val_tot / (j + 1)
                                sw.add_scalar("validation/Cycle_accent_classification_error", cycle_accent_loss_val_err,
                                              steps)

                                cycle_speaker_loss_val_err = cycle_speaker_loss_val_tot / (j + 1)
                                sw.add_scalar("validation/Cycle_speaker_classification_error", cycle_speaker_loss_val_err,
                                              steps)

                                cycle_mel_val_err = cycle_mel_val_err_tot / (j + 1)
                                sw.add_scalar("validation/Cycle_mel_spec_error", cycle_mel_val_err, steps)

                        if h.get('f0_vq_params', None):
                            sw.add_scalar("validation/commit_error", f0_commit_loss, steps)
                        if h.get('code_vq_params', None):
                            sw.add_scalar("validation/code_commit_error", code_commit_loss, steps)
                    generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    if rank == 0:
        print('Finished training')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='checkpoints/multi_loss_continue_from_pretrain_3rd_stage_cont')
    parser.add_argument('--config', default='configs/VCTK/hubert100_lut.json')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--training_steps', default=500000, type=int)
    # parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    # parser.add_argument('--checkpoint_interval', default=2500, type=int)
    # parser.add_argument('--checkpoint_interval', default=250, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    # parser.add_argument('--validation_interval', default=1000, type=int)
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
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available() and 'WORLD_SIZE' in os.environ:
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = int(os.environ['WORLD_SIZE'])
        h.batch_size = int(h.batch_size / h.num_gpus)
        local_rank = a.local_rank
        rank = a.local_rank
        print('Batch size per GPU :', h.batch_size)
    else:
        rank = 0
        local_rank = 0

    train(rank, local_rank, a, h)


if __name__ == '__main__':
    # Todo: Add numpy seed every where!!!!!!!!
    main()
