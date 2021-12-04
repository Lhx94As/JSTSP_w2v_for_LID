import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from transformer import *
from pooling_layers import *


# XSA: Baseline
# AE_XSA: FC layer-based autoencoder + XSA, multitask training
# SE_XSA: Squeeze-and-excitation (SE) layer after inputs
# AE_conv_XSA: conv layer + FC layer-based autoencoder + XSA, multitask training
# LDA_XSA: FC layers (same as the encoder in autoencoder of AE_XSA) for dimension reduction + XSA
# SE_LDA_XSA: SE layer + FC layers + XSA

class X_Transformer_E2E_LID(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=4,
                 dropout=0.1, n_lang=3, max_seq_len=10000):
        super(X_Transformer_E2E_LID, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.tdnn1 = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.fc_xv = nn.Linear(1024, feat_dim)

        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)

        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

    def mean_std_pooling(self, x, batchsize, seq_lens, weight, mask_std, weight_unb):
        """

        :param x: expect the x is of shape [Batchsize, seq_len, feature_dim]

        :param batchsize: in you script this should be len(seq_lens), namely the number of samples
        :param seq_lens: a tuple of sequence lengths
        :param weight: remove zero paddings when computing means
        :param mask_std: remove zero paddings when computing std
        :param weight_unb: do unbaised estimation, then the results are the same as x.std for fixed chunks
        :return: concatenation of means and stds
        """
        max_len = seq_lens[0]
        feat_dim = x.size(-1)
        correct_mean = x.mean(dim=1).transpose(0, 1) * weight
        correct_mean = correct_mean.transpose(0, 1)
        center_seq = x - correct_mean.repeat(1, 1, max_len).view(batchsize, -1, feat_dim)
        variance = torch.mean(torch.mul(torch.abs(center_seq) ** 2, mask_std), dim=1).transpose(0,
                                                                                                1) * weight_unb * weight
        std = torch.sqrt(variance.transpose(0, 1))
        return torch.cat((correct_mean, std), dim=1)

    def forward(self, x, seq_len, seq_weights, std_mask_, weight_unbaised, atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = self.dropout(x)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.bn3(F.relu(self.tdnn3(x)))

        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x += noise * eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        # print("pooling", stats.size())
        embedding = self.fc_xv(stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        # output, _ = self.attention_block3(output, atten_mask)
        # output, _ = self.attention_block4(output, atten_mask)
        stats = self.mean_std_pooling(output, batch_size, seq_len, seq_weights, std_mask_, weight_unbaised)
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

class AE_XSA(nn.Module):
    def __init__(self, input_dim=1024, middim=256, feat_dim=64,
                 d_k=64, d_v=64, d_ff=64, n_heads=4,
                 dropout=0.1, n_lang=3, max_seq_len=10000):
        super(AE_XSA, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.middim = middim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512),
                                     nn.Tanh(),
                                     nn.Linear(512, 256),
                                     nn.Tanh(),
                                     nn.Linear(256, 256))
        self.decoder = nn.Sequential(nn.Linear(256, 256),
                                     nn.Tanh(),
                                     nn.Linear(256, 512),
                                     nn.Tanh(),
                                     nn.Linear(512, input_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.tdnn1 = nn.Conv1d(in_channels=middim, out_channels=512, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.fc_xv = nn.Linear(1024, feat_dim)

        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)

        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

    def mean_std_pooling(self, x, batchsize, seq_lens, weight, mask_std, weight_unb):
        max_len = seq_lens[0]
        feat_dim = x.size(-1)
        correct_mean = x.mean(dim=1).transpose(0, 1) * weight
        correct_mean = correct_mean.transpose(0, 1)
        center_seq = x - correct_mean.repeat(1, 1, max_len).view(batchsize, -1, feat_dim)
        variance = torch.mean(torch.mul(torch.abs(center_seq) ** 2, mask_std), dim=1).transpose(0,
                                                                                                1) * weight_unb * weight
        std = torch.sqrt(variance.transpose(0, 1))
        return torch.cat((correct_mean, std), dim=1)

    def compute_without_padding(self, x, nonlinear):
        return torch.nn.utils.rnn.PackedSequence(nonlinear(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices)

    def forward(self, x, seq_len, seq_weights, std_mask_, weight_unbaised, atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)

        # x = x.contiguous().view( -1, self.input_dim)
        # x = rnn_utils.pack_padded_sequence(x, lengths=seq_len, batch_first=True)
        # latent_rep = torch.nn.utils.rnn.PackedSequence(self.encoder(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        # latent_rep = self.compute_without_padding(x, self.encoder)
        # ae_output = self.compute_without_padding(latent_rep, self.decoder)

        x_pack = rnn_utils.pack_padded_sequence(x, lengths=seq_len, batch_first=True)
        x_pack_size = x_pack.data.size(0)
        x_enc = x_pack.data.view(-1, self.input_dim)
        latent_rep = self.encoder(x_enc)
        output_dec = self.decoder(latent_rep)
        x = rnn_utils.PackedSequence(latent_rep.view(x_pack_size, -1),
                                     x_pack.batch_sizes,
                                     x_pack.sorted_indices,
                                     x_pack.unsorted_indices)
        # print(rnn_utils.pad_packed_sequence(x, batch_first=True)[0].size())
        x = rnn_utils.pad_packed_sequence(x, batch_first=True)[0].contiguous().\
            view(batch_size*T_len, -1, self.middim).transpose(-1, -2)

        x = self.bn1(F.relu(self.tdnn1(x)))
        x = self.bn2(F.relu(self.tdnn2(x)))
        x = self.bn3(F.relu(self.tdnn3(x)))

        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x += noise * eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        # print("pooling", stats.size())
        embedding = self.fc_xv(stats)
        embedding = embedding.view(batch_size, -1, self.feat_dim)
        # print(embedding.size())
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        # output, _ = self.attention_block3(output, atten_mask)
        # output, _ = self.attention_block4(output, atten_mask)
        stats = self.mean_std_pooling(output, batch_size, seq_len, seq_weights, std_mask_, weight_unbaised)
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output, output_dec





