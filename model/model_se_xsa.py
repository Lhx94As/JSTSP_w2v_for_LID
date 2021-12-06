from pooling_layers import *
import torch.nn.utils.rnn as rnn_utils
from transformer import *


class SE_XSA(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=4,
                 dropout=0.1, n_lang=3, max_seq_len=10000):
        super(SE_XSA, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.SElayer = SElayer_random(dim=self.input_dim)
        self.SE_clf = nn.Sequential(nn.Linear(self.input_dim//16, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, n_lang))
        self.tdnn_layers = nn.Sequential(nn.Dropout(p=dropout),
                                         nn.Conv1d(in_channels=self.input_dim, out_channels=512, kernel_size=5, dilation=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=False),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=False),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=False))

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

    def forward(self, x, seq_len, seq_weights, std_mask_, weight_unbaised, atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = x.contiguous().view(batch_size, -1, self.input_dim)
        x, se_, se_mid = self.SElayer(x, seq_weights) # x: (batchsize, frames, input_dim)
        se_prediction = self.SE_clf(se_mid)
        x = x.contiguous().view(batch_size*T_len, -1, self.input_dim).transpose(1,2)
        x = self.tdnn_layers(x)

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
        stats = self.mean_std_pooling(output, batch_size, seq_len, seq_weights, std_mask_, weight_unbaised)
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output, se_prediction, se_
