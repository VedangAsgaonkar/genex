import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import sys
import gc


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        print(x.shape)
        return x.permute(0, 2, 1)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Configuration():
    def __init__(self, hidden_size, num_layers, attention_dropout_rate, num_heads, latent_size):
        self.hidden_size = hidden_size
        self.transformer = {
            'num_layers' : num_layers,
            'attention_dropout_rate' : attention_dropout_rate,
            'num_heads' : num_heads,
            'mlp_dim' : 3,
            'dropout_rate' : attention_dropout_rate,
        }
        self.latent_size = latent_size

class Transformer_Encoder(nn.Module):
    def __init__(self, config, vis=False):
        super(Transformer_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.fc = Linear(config.hidden_size, 2*config.latent_size)

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        latent = self.fc(encoded)

        return encoded, latent

class Set_Embedding(nn.Module):
    def __init__(self, set_size, hidden_size):
        super(Set_Embedding, self).__init__()
        self.fc = nn.Linear(set_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, mask, free_mem=False, eval=False):
        # bs = 1
        # x is a vector of size [batch_size, set_size]
        # mask is a bit vector of size [batch_size, set_size]
        # [batchsize*num of ones, set_size]
        # print("x shape", x.shape, mask.shape)
        # print(torch.sum(mask, dim=1))
        # print("set 0",torch.cuda.memory_allocated(4))
        # eye = torch.eye(x.shape[1]).repeat(x.shape[0],1,1)
        # one_hot = eye[torch.gt(mask,0)]
        one_hot = torch.eye(x.shape[1])[torch.nonzero(mask)[:,1]]
        # print("oh", sys.getsizeof(one_hot.storage()), one_hot.device)
        # print("set 1",torch.cuda.memory_allocated(4))
        emb = self.fc(one_hot)
        # print("set 2",torch.cuda.memory_allocated(4))
        # print("emb", emb.shape)
        y =  x[torch.gt(mask,0)].unsqueeze(1)
        # print("set 3",torch.cuda.memory_allocated(4))
        out = torch.hstack([emb, y])
        # print("out", out.shape)
        num_bits_set = torch.sum(mask[0])
        # print("num_bits_set", num_bits_set)
        # out = out.reshape((x.shape[0], int(num_bits_set.item()), x.shape[1]+1))
        out = out.reshape((x.shape[0], int(num_bits_set.item()), self.hidden_size+1))
        # out = torch.cat([emb, x[torch.gt(mask,0)[0]][:,None]], dim=1)
        # print("set 4",torch.cuda.memory_allocated(4))
        if free_mem:
            one_hot = one_hot.cpu()
            del one_hot
            # print("set 5",torch.cuda.memory_allocated(4))
        return out.float()


def VAE_sample(encoder, x, latent_size, eval=False):
    encoded, latent = encoder(x, free_mem=eval)
    mean = latent[:,:latent_size]
    logvar = latent[:,latent_size:]
    stddev = torch.sqrt(torch.exp(logvar))
    epsilon = torch.randn((x.shape[0], latent_size))
    if eval:
        encoded = encoded.cpu()
        del encoded
        latent = latent.cpu()
        del latent
    return mean+stddev*epsilon, mean, logvar

def ELBO_loss(output, target, present_mask, mu, logvar, args):
    if args.data == 'disease_pred':
        ce = nn.BCELoss()((output*present_mask).float(), (target*present_mask).float())
    elif args.data == 'covid' or args.data == 'proteomic' or args.data=='mnist':
        std = 0.1*np.sqrt(2)*np.sqrt(784)
        # print(output[0], target[0])
        ce = torch.sum(0.5 * (target - output)**2 / std**2)
    else:
        raise NotImplementedError
    kl = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp())
    # print("ce", ce, "kl", kl)
    # return ce+kl
    return ce


class Transformer_Encoder_pytorch(nn.Module):

    # def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                #  nlayers: int, dropout: float = 0.5):
    def __init__(self, args):
        super().__init__()
        self.model_type = 'Transformer'
        # self.pos_encoder = PositionalEncoding(args.d_model, dropout)
        encoder_layers = TransformerEncoderLayer(args.encoder_output_dim, args.num_heads, args.encoder_output_dim, args.attention_dropout_rate, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.num_layers)
        self.fc = Linear(args.encoder_output_dim, 2*args.latent_dim)
        # self.encoder = nn.Embedding(args.ntoken, d_model)
        # self.d_model = d_model
        # self.decoder = nn.Linear(d_model, ntoken)

        # self.init_weights()

    # def init_weights(self) -> None:
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask = None, free_mem=False) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        # torch.cuda.synchronize()
        # print("trans 1",torch.cuda.memory_allocated(4))
       
        ones = torch.ones((src.shape[1], src.shape[1]))
        # print("trans 1.1",torch.cuda.memory_allocated(4))
        encoded = self.transformer_encoder(src, ones)
        
        # print("trans 1.2",torch.cuda.memory_allocated(4))
        encoded = torch.mean(encoded, axis=1)
        
        # print("encoded:", sys.getsizeof(encoded.storage()))
        # print("ones:", sys.getsizeof(ones.storage()))
        if free_mem:
            src = src.cpu()
            del src
            ones = ones.cpu()
            del ones
        latent = self.fc(encoded)
        # output = self.decoder(output)
        return encoded, latent


# def generate_square_subsequent_mask(sz: int) -> Tensor:
#     """Generates an upper-triangular matrix of -inf, with zeros on diag."""
#     return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)