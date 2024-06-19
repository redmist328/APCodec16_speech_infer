import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from utils import init_weights, get_padding
import numpy as np
from quantize import ResidualVectorQuantize

LRELU_SLOPE = 0.1


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value= None,
        adanorm_num_embeddings = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id = None) :
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()

        self.input_channels=h.n_fft//2+1
        self.h=h
        self.dim=256
        self.num_layers=8
        self.adanorm_num_embeddings=None
        self.intermediate_dim=512
        self.embed_logamp = nn.Conv1d(self.input_channels, self.dim, kernel_size=7, padding=3)
        self.embed_pha = nn.Conv1d(self.input_channels, self.dim, kernel_size=7, padding=3)
        self.norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.norm_pha = nn.LayerNorm(self.dim, eps=1e-6)
        layer_scale_init_value =  1 / self.num_layers
        self.convnext_logamp = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.convnext_pha = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer_norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.final_layer_norm_pha = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)
        #out_dim = h.n_fft + 2
        self.out_logamp = torch.nn.Linear(self.dim, h.AMP_Encoder_channel)
        self.out_pha = torch.nn.Linear(self.dim, h.PHA_Encoder_channel)

        self.AMP_Encoder_downsample_output_conv = weight_norm(Conv1d(h.AMP_Encoder_channel, h.AMP_Encoder_channel//2, 15, h.ratio, 
                                                  padding=get_padding(15, 1)))
        self.PHA_Encoder_downsample_output_conv = weight_norm(Conv1d(h.PHA_Encoder_channel, h.PHA_Encoder_channel//2, 15, h.ratio, 
                                                  padding=get_padding(15, 1)))

        self.latent_output_conv = weight_norm(Conv1d(h.AMP_Encoder_channel//2+h.PHA_Encoder_channel//2, h.latent_dim, h.latent_output_conv_kernel_size, 1, 
                                                  padding=get_padding(h.latent_output_conv_kernel_size, 1)))

        self.AMP_Encoder_downsample_output_conv.apply(init_weights)
        self.PHA_Encoder_downsample_output_conv.apply(init_weights)
        self.latent_output_conv.apply(init_weights)


        self.quantizer = ResidualVectorQuantize(
            input_dim=h.latent_dim,
            codebook_dim=h.latent_dim,
            n_codebooks=h.n_codebooks,
            codebook_size=1024,
            quantizer_dropout=False
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, logamp, pha):

        logamp_encode = self.embed_logamp(logamp)
        logamp_encode = self.norm_logamp(logamp_encode.transpose(1, 2))
        logamp_encode = logamp_encode.transpose(1, 2)
        for conv_block in self.convnext_logamp:
            logamp_encode = conv_block(logamp_encode, cond_embedding_id=None)
        logamp_encode = self.final_layer_norm_logamp(logamp_encode.transpose(1, 2))
        logamp_encode = self.out_logamp(logamp_encode).transpose(1, 2)
        logamp_encode = self.AMP_Encoder_downsample_output_conv(logamp_encode)

        pha_encode = self.embed_pha(pha)
        pha_encode = self.norm_pha(pha_encode.transpose(1, 2))
        pha_encode = pha_encode.transpose(1, 2)
        for conv_block in self.convnext_pha:
            pha_encode = conv_block(pha_encode, cond_embedding_id=None)
        pha_encode = self.final_layer_norm_pha(pha_encode.transpose(1, 2))
        pha_encode = self.out_pha(pha_encode).transpose(1, 2)
        pha_encode = self.PHA_Encoder_downsample_output_conv(pha_encode)

        encode = torch.cat((logamp_encode, pha_encode), -2) 

        latent = self.latent_output_conv(encode)
        latent,codes,_,commitment_loss,codebook_loss = self.quantizer(latent)
        return latent,codes,commitment_loss,codebook_loss


class Decoder(torch.nn.Module):
    def __init__(self, h):
        super(Decoder, self).__init__()

        self.h=h
        self.dim=256
        self.num_layers=8
        self.adanorm_num_embeddings=None
        self.intermediate_dim=512

        self.latent_input_conv = weight_norm(Conv1d(h.latent_dim, h.AMP_Decoder_channel//2+h.PHA_Decoder_channel//2, h.latent_input_conv_kernel_size, 1, 
                                                 padding=get_padding(h.latent_input_conv_kernel_size, 1)))

        self.AMP_Decoder_upsample_input_conv = weight_norm(ConvTranspose1d(h.AMP_Decoder_channel//2+h.PHA_Decoder_channel//2, h.AMP_Decoder_channel,
                                                           h.AMP_Decoder_input_upconv_kernel_size, h.ratio, padding=(h.AMP_Decoder_input_upconv_kernel_size-h.ratio)//2))
        self.PHA_Decoder_upsample_input_conv = weight_norm(ConvTranspose1d(h.AMP_Decoder_channel//2+h.PHA_Decoder_channel//2, h.PHA_Decoder_channel,
                                                           h.PHA_Decoder_input_upconv_kernel_size, h.ratio, padding=(h.PHA_Decoder_input_upconv_kernel_size-h.ratio)//2))


        self.embed_logamp = nn.Conv1d(h.AMP_Decoder_channel, self.dim, kernel_size=7, padding=3)
        self.embed_pha = nn.Conv1d(h.PHA_Decoder_channel, self.dim, kernel_size=7, padding=3)
        self.norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.norm_pha = nn.LayerNorm(self.dim, eps=1e-6)
        layer_scale_init_value =  1 / self.num_layers
        self.convnext_logamp = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.convnext_pha = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=self.dim,
                    intermediate_dim=self.intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer_norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.final_layer_norm_pha = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)
        #out_dim = h.n_fft + 2
        self.out_logamp = torch.nn.Linear(self.dim, h.AMP_Encoder_channel)
        self.out_pha = torch.nn.Linear(self.dim, h.PHA_Encoder_channel)

        self.AMP_Decoder_output_conv = weight_norm(Conv1d(h.AMP_Decoder_channel, h.n_fft//2+1, h.AMP_Decoder_output_conv_kernel_size, 1, 
                                                  padding=get_padding(h.AMP_Decoder_output_conv_kernel_size, 1)))
        self.PHA_Decoder_output_R_conv = weight_norm(Conv1d(h.PHA_Decoder_channel, h.n_fft//2+1, h.PHA_Decoder_output_R_conv_kernel_size, 1, 
                                                    padding=get_padding(h.PHA_Decoder_output_R_conv_kernel_size, 1)))
        self.PHA_Decoder_output_I_conv = weight_norm(Conv1d(h.PHA_Decoder_channel, h.n_fft//2+1, h.PHA_Decoder_output_I_conv_kernel_size, 1, 
                                                    padding=get_padding(h.PHA_Decoder_output_I_conv_kernel_size, 1)))

        self.AMP_Decoder_output_conv.apply(init_weights)
        self.PHA_Decoder_output_R_conv.apply(init_weights)
        self.PHA_Decoder_output_I_conv.apply(init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, latent):

        latent = self.latent_input_conv(latent)
        logamp = self.AMP_Decoder_upsample_input_conv(latent)
        logamp = self.embed_logamp(logamp)
        logamp = self.norm_logamp(logamp.transpose(1, 2))
        logamp = logamp.transpose(1, 2)
        for conv_block in self.convnext_logamp:
            logamp = conv_block(logamp, cond_embedding_id=None)
        logamp = self.final_layer_norm_logamp(logamp.transpose(1, 2))
        logamp = self.out_logamp(logamp).transpose(1, 2)
        logamp = self.AMP_Decoder_output_conv(logamp)

        pha = self.PHA_Decoder_upsample_input_conv(latent)
        pha = self.embed_pha(pha)
        pha = self.norm_pha(pha.transpose(1, 2))
        pha = pha.transpose(1, 2)
        for conv_block in self.convnext_pha:
            pha = conv_block(pha, cond_embedding_id=None)
        pha = self.final_layer_norm_pha(pha.transpose(1, 2))
        pha = self.out_pha(pha).transpose(1, 2)  
        R = self.PHA_Decoder_output_R_conv(pha)
        I = self.PHA_Decoder_output_I_conv(pha)

        pha = torch.atan2(I,R)

        rea = torch.exp(logamp)*torch.cos(pha)
        imag = torch.exp(logamp)*torch.sin(pha)

        spec = torch.cat((rea.unsqueeze(-1),imag.unsqueeze(-1)),-1)

        audio = torch.istft(spec, self.h.n_fft, hop_length=self.h.hop_size, win_length=self.h.win_size, window=torch.hann_window(self.h.win_size).to(latent.device), center=True)

        return logamp, pha, rea, imag, audio.unsqueeze(1)


