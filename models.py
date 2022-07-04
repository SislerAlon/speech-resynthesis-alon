# adapted from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from modules.jukebox import Encoder, Decoder
from utils import init_weights, get_padding, AttrDict
from modules.vq import Bottleneck


import pandas as pd
import numpy as np

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                        padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                   padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                                       padding=get_padding(kernel_size, dilation[0]))), weight_norm(
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class VariancePredictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim,
        var_pred_hidden_dim,
        var_pred_kernel_size,
        var_pred_dropout
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim, var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size,
                padding=(var_pred_kernel_size - 1) // 2
            ),
            nn.ReLU()
        )
        self.ln1 = nn.LayerNorm(var_pred_hidden_dim)
        self.dropout = var_pred_dropout
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                var_pred_hidden_dim, var_pred_hidden_dim,
                kernel_size=var_pred_kernel_size, padding=1
            ),
            nn.ReLU()
        )
        self.ln2 = nn.LayerNorm(var_pred_hidden_dim)
        self.proj = nn.Linear(var_pred_hidden_dim, 1)

    def forward(self, x):
        # Input: B x T x C; Output: B x T
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln1(x), p=self.dropout, training=self.training)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(self.ln2(x), p=self.dropout, training=self.training)
        return self.proj(x).squeeze(dim=2)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(getattr(h, "model_in_dim", 128), h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)), k,
                                u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

def process_duration(code, code_feat):
    uniq_code_count = []
    uniq_code_feat = []
    for i in range(code.size(0)):
        _, count = torch.unique_consecutive(code[i, :], return_counts=True)
        if len(count) > 2:
            # remove first and last code as segment sampling may cause incomplete segment length
            uniq_code_count.append(count[1:-1])
            uniq_code_idx = count.cumsum(dim=0)[:-2]
        else:
            uniq_code_count.append(count)
            uniq_code_idx = count.cumsum(dim=0) - 1
        uniq_code_feat.append(code_feat[i, uniq_code_idx, :].view(-1, code_feat.size(2)))
    uniq_code_count = torch.cat(uniq_code_count)

    # collate feat
    max_len = max(feat.size(0) for feat in uniq_code_feat)
    out = uniq_code_feat[0].new_zeros((len(uniq_code_feat), max_len, uniq_code_feat[0].size(1)))
    mask = torch.arange(max_len).repeat(len(uniq_code_feat), 1)
    for i, v in enumerate(uniq_code_feat):
        out[i, : v.size(0)] = v
        mask[i, :] = mask[i, :] < v.size(0)

    return out, mask.bool(), uniq_code_count.float()



class CodeGenerator(Generator):
    def __init__(self, h):
        super().__init__(h)
        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)
        self.f0 = h.get('f0', None)
        self.multispkr = h.get('multispkr', None)

        if self.multispkr:
            # without pre-train
            # self.spkr = nn.Embedding(200, h.embedding_dim)
            with open(r"src/x_vectors_embedding/represenation_x_vectors.npy", 'rb') as f:
                speaker_weight_matrix = np.load(f)

            num_embeddings_speaker, embedding_dim_speaker = speaker_weight_matrix.shape
            self.spkr = nn.Embedding(num_embeddings_speaker, embedding_dim_speaker)
            self.spkr.weight = torch.nn.Parameter(torch.from_numpy(speaker_weight_matrix).float(), requires_grad=False)
            self.spkr.weight.requires_grad = False

            # self.spkr = nn.Embedding(200, h.embedding_dim // 2)

            self.Accent_embedding_by_accent = h.get('Accent_embedding_by_accent', None)

            if self.Accent_embedding_by_accent:
                # with open(r"C:\git\Paccent_classifier\accent_represenation_x_vectors.npy", 'rb') as f:
                with open(r"src/accent_model/accent_represenation_x_vectors.npy", 'rb') as f:
                    weight_matrix = np.load(f)
            else: # by speaker - diff aceent emdedding per speaker
                with open(r"src/represenation_mfcc/represenation_mfcc.npy", 'rb') as f:
                    weight_matrix = np.load(f)

            speakers_data_path = r"src/speakers_data.csv"
            speakers_data = pd.read_csv(speakers_data_path).set_index('Id')
            self.speakers_data_dict = speakers_data.to_dict('index')

            num_embeddings, embedding_dim = weight_matrix.shape
            self.accent_lt = nn.Embedding(num_embeddings, embedding_dim)
            self.accent_lt.weight = torch.nn.Parameter(torch.from_numpy(weight_matrix).float(), requires_grad=False)
            self.accent_lt.weight.requires_grad = False
            self.accent = True

            assert embedding_dim+embedding_dim_speaker == 128

        self.encoder = None
        self.vq = None
        if h.get("lambda_commit", None):
            assert self.f0, "Requires F0 set"
            self.encoder = Encoder(**h.f0_encoder_params)
            self.vq = Bottleneck(**h.f0_vq_params)

        self.code_encoder = None
        self.code_vq = None
        if h.get('lambda_commit_code', None):
            self.code_encoder = Encoder(**h.code_encoder_params)
            self.code_vq = Bottleneck(**h.code_vq_params)
            self.dict = None

        self.quantizer = None
        if h.get('f0_quantizer_path', None):
            assert self.f0, "Requires F0 set"
            self.quantizer = Quantizer(AttrDict(h.f0_quantizer))
            quantizer_state = torch.load(h.f0_quantizer_path, map_location='cpu')
            self.quantizer.load_state_dict(quantizer_state['generator'])
            self.quantizer.eval()
            self.f0_dict = nn.Embedding(h.f0_quantizer['f0_vq_params']['l_bins'], h.embedding_dim)

        self.dur_predictor = None
        if h.get('dur_prediction_weight', None):
            self.dur_predictor = VariancePredictor(**h.dur_predictor_params)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError('Padding condition signal - misalignment between condition features.')

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        code_commit_losses = None
        code_metrics = None
        if self.code_vq and kwargs['code'].dtype is torch.int64:
            x = self.code_vq.level_blocks[0].k[kwargs['code']].transpose(1, 2)
        elif self.code_vq:
            code_h = self.code_encoder(kwargs['code'])
            _, code_h_q, code_commit_losses, code_metrics = self.code_vq(code_h)
            x = code_h_q[0]
        else:
            x = self.dict(kwargs['code']).transpose(1, 2)

#############################################
        dur_losses = 0.0
        if self.dur_predictor:
            if self.training:
                # assume input code is always full sequence
                uniq_code_feat, uniq_code_mask, dur = process_duration(
                    kwargs['code'], x.transpose(1, 2))
                log_dur_pred = self.dur_predictor(uniq_code_feat)
                log_dur_pred = log_dur_pred[uniq_code_mask]
                log_dur = torch.log(dur + 1)
                dur_losses = F.mse_loss(log_dur_pred, log_dur, reduction="mean")
            elif kwargs.get('dur_prediction', False):
                # assume input code can be unique sequence only in eval mode
                assert x.size(0) == 1, "only support single sample batch in inference"
                log_dur_pred = self.dur_predictor(x.transpose(1, 2))
                dur_out = torch.clamp(
                    torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
                )
                # B x C x T
                x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)
###################################################


        f0_commit_losses = None
        f0_metrics = None
        if self.vq:
            f0_h = self.encoder(kwargs['f0'])
            _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
            kwargs['f0'] = f0_h_q[0]
        elif self.quantizer:
            self.quantizer.eval()
            assert not self.quantizer.training, "VQ is in training status!!!"
            f0_h = self.quantizer.encoder(kwargs['f0'])
            f0_h = [x.detach() for x in f0_h]
            zs, _, _, _ = self.quantizer.vq(f0_h)
            zs = [x.detach() for x in zs]
            f0_h_q = self.f0_dict(zs[0].detach()).transpose(1, 2)
            kwargs['f0'] = f0_h_q

        if self.f0:
            if x.shape[-1] < kwargs['f0'].shape[-1]:
                x = self._upsample(x, kwargs['f0'].shape[-1])
            else:
                kwargs['f0'] = self._upsample(kwargs['f0'], x.shape[-1])
            x = torch.cat([x, kwargs['f0']], dim=1)

        if self.multispkr:
            spkr = self.spkr(kwargs['spkr']).transpose(1, 2)
            # spkr = self.spkr(torch.tensor([[30]],dtype=torch.int32).cuda()).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        ####### Accent code ########
        if self.accent:
            if 'accent_id' in kwargs:
                id_to_take = kwargs['accent_id']
            else:
                print("Are you sure????")
                id_to_take = kwargs['spkr']
            accent = self.accent_lt(id_to_take).transpose(1, 2)
            # accent = self.accent_lt(torch.tensor([[107]],dtype=torch.int32).cuda()).transpose(1, 2)
            accent = self._upsample(accent, x.shape[-1])
            x = torch.cat([x, accent], dim=1)
        ############################

        for k, feat in kwargs.items():
            # if k in ['spkr', 'code', 'f0']:
            if k in ['spkr', 'code', 'f0', 'accent_id']:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        if self.vq or self.code_vq:
            return super().forward(x), (code_commit_losses, f0_commit_losses), (code_metrics, f0_metrics)
        else:
            return super().forward(x), dur_losses


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
             norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))), ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(2), DiscriminatorP(3), DiscriminatorP(5), DiscriminatorP(7), DiscriminatorP(11), ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
             norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
             norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
             norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
             norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2)), ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS(), ])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Quantizer(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.encoder = Encoder(**h.f0_encoder_params)
        self.vq = Bottleneck(**h.f0_vq_params)
        self.decoder = Decoder(**h.f0_decoder_params)

    def forward(self, **kwargs):
        f0_h = self.encoder(kwargs['f0'])
        _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
        f0 = self.decoder(f0_h_q)

        return f0, f0_commit_losses, f0_metrics


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
