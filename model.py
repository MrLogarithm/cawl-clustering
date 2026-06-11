import torch
import math
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :].squeeze()
        return self.dropout(x)

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim

        self.dropout= nn.Dropout(0.5)
        self.eConv1 = nn.Conv2d(1,3,8) 
        self.eConv2 = nn.Conv2d(3,6,8)
        self.ePool1 = nn.MaxPool2d(3,3)
        self.eConv3 = nn.Conv2d(6,8,8)
        self.ePool2 = nn.MaxPool2d(3,3)
        self.eF1    = nn.Linear(72,z_dim)

        self.eMu    = nn.Linear(z_dim, z_dim)
        self.eSigma = nn.Linear(z_dim, z_dim)

        self.dConvT1     = nn.ConvTranspose2d(z_dim, 60,8) # 1 > 8
        self.dBatchNorm1 = nn.BatchNorm2d( 60)
        self.dConvT2     = nn.ConvTranspose2d( 60, 30,8,2) # > 22
        self.dBatchNorm2 = nn.BatchNorm2d( 30)
        self.dConvT3     = nn.ConvTranspose2d( 30,15,8,2) # > 50
        self.dBatchNorm3 = nn.BatchNorm2d(15)
        self.dConvT4     = nn.ConvTranspose2d(15,1,15,1) # > 64

    def encode(self,x):
        # x = self.dropout(x)
        x = self.eConv1(x)
        x = F.relu(x)
        x = self.eConv2(x)
        x = F.relu(x)
        x = self.ePool1(x)
        x = self.eConv3(x)
        x = F.relu(x)
        x = self.ePool2(x)
        x = x.view(x.size()[0], -1)
        x = self.eF1(x)

        return x

    def project(self, x):
        mu    = self.eMu(x)
        sigma = self.eSigma(x)

        return (mu, sigma)

    # From https://github.com/pytorch/examples/blob/master/vae/main.py
    def reparametrize(self, mu, sigma):
        std = torch.exp(0.5*sigma)
        eps = torch.randn_like(std)

        z = mu + eps*std
        
        return z
        
    def decode(self,x):
        x = torch.reshape(x,(x.shape[0],self.z_dim,1,1))
        x = self.dConvT1(x)
        x = self.dBatchNorm1(x)
        x = F.relu(x)
        x = self.dConvT2(x)
        x = self.dBatchNorm2(x)
        x = F.relu(x)
        x = self.dConvT3(x)
        x = self.dBatchNorm3(x)
        x = F.relu(x)
        x = self.dConvT4(x)
        x = torch.sigmoid(x)

        return x

class TransformerLM(nn.Module):
    def __init__(self, z_dim):
        super(TransformerLM, self).__init__()
        self.z_dim = z_dim

        self.ePos = PositionalEncoding(z_dim,0.5)
        self.eEmb = nn.Linear(z_dim,z_dim)
        self.eLM = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    z_dim, # input dimension
                    8,
                    z_dim, # output dimension
                    0.5,
                    batch_first = True,
                ),
                6, # num. layers
        )

    def forward(self, x, mask_prob=0.05):
        masked = x.clone()
        # replace masked tokens with random normal noise
        mask   = torch.bernoulli(mask_prob*torch.ones_like(x[:,0])).bool()
        masked[mask].uniform_()
        emb = self.eEmb(masked)
        pos = self.ePos(emb)
        y = self.eLM(
            pos,
            #mask=(1 - torch.tril(torch.ones(pos.shape[0],pos.shape[0]))).bool().to("cuda")
            mask=torch.eye(pos.shape[0]).bool().to("cuda")
        )
        return y

class LSTMLM(nn.Module):
    def __init__(self, z_dim):
        super(LSTMLM, self).__init__()
        self.eLM = nn.LSTM(
            z_dim,
            z_dim,
            num_layers=1,
            bidirectional=False,
        )

    def forward(self, x):
        y, _ = self.eLM(x)
        return y

class VAECluster(nn.Module):
    def __init__(self, expt, H_DIM):
        super(VAECluster, self).__init__()
        self.expt = expt
        self.H_DIM = H_DIM

        self.vae  = VAE(self.H_DIM)
        if expt.gen_from_txr:
            self.txr  = TransformerLM(self.H_DIM)
        if expt.gen_from_lstm:
            self.lstm = LSTMLM(self.H_DIM)

        if expt.gen_from_context:
            # combine neighbors
            self.merge = nn.Linear(2*self.H_DIM, self.H_DIM)

        if expt.gen_from_txr or expt.gen_from_lstm:
            # decode LM output
            self.dLM = nn.Linear(self.H_DIM, self.H_DIM)

            if expt.reparametrize_lm:
                # decode LM output to mu, sigma
                self.lmMu = nn.Linear(self.H_DIM,self.H_DIM)
                self.lmSigma = nn.Linear(self.H_DIM,self.H_DIM)

    def forward(self,x, mask_prob=0.05):
        results = dict()
        results["vae_mu"], results["vae_sigma"] = self.vae.project(
            self.vae.encode(x)
        )
        results["vae_z"] = self.vae.reparametrize(
            results["vae_mu"],
            results["vae_sigma"],
        )

        if self.expt.gen_from_self:
            results["gen_vae_self"] = self.vae.decode(results["vae_z"])

        if self.expt.gen_from_context:
            lt_ctx  = results["vae_z"][:-2]
            rt_ctx  = results["vae_z"][2:]
            context = torch.cat([lt_ctx, rt_ctx], dim=1)
            context = self.merge(context)
            if self.expt.reparametrize_merged:
                context_mu, context_sigma = self.vae.project(context)
                context = self.vae.reparametrize(context_mu, context_sigma)
            results["gen_vae_context"] = self.vae.decode(context)

        # Transformer
        if self.expt.gen_from_txr:
            results["txr_y"] = self.txr(results["vae_z"], mask_prob=mask_prob)
            # TODO do this before or after the reparametrization?
            results["txr_z"] = self.dLM(results["txr_y"])
            if self.expt.reparametrize_lm:
                # TODO try using the same projection as the VAE
                results["txr_mu"]    = self.lmMu(results["txr_z"])
                results["txr_sigma"] = self.lmSigma(results["txr_z"])
                results["txr_z"]     = self.vae.reparametrize(
                    results["txr_mu"], 
                    results["txr_sigma"]
                )

            results["gen_txr"] = self.vae.decode(results["txr_z"])

        # LSTM
        if self.expt.gen_from_lstm:
            results["lstm_y"] = self.lstm(results["vae_z"])
            # TODO do this before or after the reparametrization?
            results["lstm_z"] = self.dLM(results["lstm_y"])
            if self.expt.reparametrize_lm:
                # TODO try using the same projection as the VAE
                results["lstm_mu"]    = self.lmMu(results["lstm_z"])
                results["lstm_sigma"] = self.lmSigma(results["lstm_z"])
                results["lstm_z"]     = self.vae.reparametrize(
                    results["lstm_mu"], 
                    results["lstm_sigma"]
                )

            results["gen_lstm"] = self.vae.decode(results["lstm_z"])

        return results
