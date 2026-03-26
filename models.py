from functions import ReverseLayerF

import torch.nn as nn
import torch
import torch.nn.functional as F



from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pywt

class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        #print(x.shape)
        #print(adj.shape)
        out = []
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        #print(output.shape)
        return output
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention——> [B, N, N]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = ModuleList(
                [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x    
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input




class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        print('x:', x.shape)
        # x: [B, L, D, 4]
        # Assume: x[..., 1] = time-of-day (normalized 0–1)
        #         x[..., 2] = day-of-week (0–6)
        B, L, D, _ = x.shape

        # --- Time of day ---
        day_emb = x[..., 1]  # [B, L, D]
        day_idx = (day_emb * self.time).long()  # scale to [0, time-1]
        day_idx = day_idx.clamp(0, self.time - 1)  # ensure safe indexing
        # print('d:', day_idx.shape)
        # # exit()
        time_day = self.time_day[day_idx]  # [B, L, D, features]
        
        time_day = time_day.transpose(1, 2).contiguous()  # [B, D, L, features]

        # --- Day of week ---
        week_emb = x[..., 2]  # [B, L, D]
        week_idx = week_emb.long().clamp(0, 6)
        time_week = self.time_week[week_idx]  # [B, L, D, features]
        time_week = time_week.transpose(1, 2).contiguous()

        # --- Combine ---
        tem_emb = time_day + time_week  # [B, D, L, features]
        tem_emb = tem_emb.permute(0, 3, 1, 2)  # [B, features, D, L]
        print('tem_emb:', tem_emb.shape)
        # exit()
        return tem_emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        print('num_nodes:', num_nodes)
        
        self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        # nn.init.xavier_uniform_(self.b)
        self.bn = nn.BatchNorm1d(tem_size)

    def forward(self, seq):
                
        seq = seq.transpose(3, 2)
        
        seq = seq.permute(0, 1, 3, 2).contiguous()
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n
        
        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(axis=1)  # b,c,n  [50, 1, 12]
        matmul_out=torch.matmul(f1, self.w)
        out2=torch.matmul(matmul_out, f2)
        
        #add out2+self.b
        logits = torch.sigmoid(out2 + self.b)
        
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        
        logits = self.bn(logits).permute(0, 2, 1).contiguous()



        coefs = torch.softmax(logits, -1)
        T_coef = coefs.transpose(-1, -2)

        x_1 = torch.einsum('bcnl,blq->bcnq', seq, T_coef)
        
        return x_1

class DualChannelLearner(nn.Module):
    def __init__(self, features=128, layers=4, length=12, num_nodes=170, dropout=0.1):
        super(DualChannelLearner, self).__init__()


        self.low_freq_layers = nn.ModuleList([
            TATT_1(features, num_nodes, length) for _ in range(layers)
        ])


        kernel_size = int(length / layers + 1)
        dilation_size = None
        self.high_freq_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    features, 
                    features, 
                    (1, kernel_size)),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(layers)
        ])

        self.alpha = nn.Parameter(torch.tensor(-5.0))
        self.conv_fusion = nn.Conv2d(in_channels=features,
                                    out_channels=features,
                                    kernel_size=(1, 1))
    def forward(self, XL, XH):
        res_xl = XL
        res_xh = XH

        for layer in self.low_freq_layers:
            XL = layer(XL)
        XL = XL + res_xl


        XH = nn.functional.pad(XH, (1, 0, 0, 0))  
        for layer in self.high_freq_layers:
            XH = layer(XH)  

        XH = XH + res_xh
        
        alpha_sigmoid = torch.sigmoid(self.alpha)
        output = alpha_sigmoid * XL + (1 - alpha_sigmoid) * XH
        
        output = self.conv_fusion(output)

        return output 


    
class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output

 
class HGL_layer(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(HGL_layer, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model

        
        self.gcn = Diffusion_GCN(channels=256, diffusion_step=1, dropout=dropout)
        self.gat = GAT(256, 256, dropout, alpha=0.2, nheads=1)
        
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        self.alpha = nn.Parameter(torch.tensor(-5.0)) 
        self.weight = nn.Parameter(torch.ones(256, self.num_nodes, 1))   
        self.bias = nn.Parameter(torch.zeros(256, self.num_nodes, 1))
        
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 6).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(6, num_nodes).to(device), requires_grad=True).to(device)
        

    def forward(self, input, D_Graph):
        #print('input', input.shape)        #input torch.Size([64, 256, 170, 1])        

        A_graph = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1).unsqueeze(0)
        x_gcn = self.gcn(input, A_graph)

        x_gat = self.gat(input.transpose(1,3), D_Graph).transpose(1,3)


        alpha_sigmoid = torch.sigmoid(self.alpha)  
        x =  alpha_sigmoid* x_gat +  (1 - alpha_sigmoid) * x_gcn
        
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * self.weight + self.bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        

        return x
    

class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, length=12, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.E_s = nn.Parameter(torch.randn(channels, num_nodes))
        self.length = length
        self.E_tod = nn.Parameter(torch.randn(1, channels, 1, length))
        self.E_dow = nn.Parameter(torch.randn(1, channels, 1, length))
        nn.init.xavier_uniform_(self.E_s)
        nn.init.xavier_uniform_(self.E_tod)
        nn.init.xavier_uniform_(self.E_dow)
        self.fc = nn.Linear(2,1)
        
    def forward(self, x):
        #E_t = X*E_w*E_d
        
        E_t = x*self.E_tod*self.E_tod

        E_d = torch.tanh(
            torch.einsum("bcnt, cm->bnm", x, self.E_s).contiguous()
        )

        A_adp = torch.softmax(
            F.relu(
                torch.einsum("bnc, bmc->bnm", E_d, E_d).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        
        return (A_adp.mean(dim=0) > 0.5).float()



class HybridGraphLearner(nn.Module):
    def __init__(self, channels=128, layers = 2, num_nodes=170, length=128, diffusion_step=1, dropout=0.1, emb=None, device=None):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.layers = layers
        self.generator = Graph_Generator(channels, num_nodes, length, diffusion_step, dropout)
        
        self.gcn = nn.ModuleList([
            Diffusion_GCN(channels, 
                          diffusion_step, 
                          dropout)
            for _ in range(self.layers)
        ])
        self.gat = nn.ModuleList([
            GAT(channels, 
                channels, 
                dropout, 
                alpha=0.2, 
                nheads=1)
            for _ in range(self.layers)
        ])
        
        self.device = device
        self.alpha = nn.Parameter(torch.tensor(-5.0)) 
        self.emb =  nn.Parameter(torch.randn(channels, num_nodes, 12))
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 6).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(6, num_nodes).to(device), requires_grad=True).to(device)
        
        self.conv_gcn = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        
        self.conv_gat = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        
        self.conv_fusion = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        self.alpha = nn.Parameter(torch.tensor(-5.0))
    def forward(self, x):
        print('x:', x.shape)
        
        A_adp = F.softmax(
                    F.relu(torch.mm(self.nodevec1, self.nodevec2)),
                    dim=1).unsqueeze(0)
        skip = x
        x = self.conv(x)
        A_dyn = self.generator(x)
        print('A_dyn:', A_dyn.shape)
        exit()
        for gcn_layer in self.gcn:
            x_gcn = gcn_layer(x, A_adp) 
        x_gcn = self.conv_gcn(x_gcn)
            #print(x_gcn.shape)
        for gat_layer in self.gat:
            x_gat = gat_layer(x.transpose(1, 3), A_dyn).transpose(1, 3)
        x_gat = self.conv_gat(x_gat)
        #spatial gate
        alpha_sigmoid = torch.sigmoid(self.alpha)
        x = alpha_sigmoid * x_gcn + (1 - alpha_sigmoid) * x_gat
        x = self.conv_fusion(x)
        x = x*self.emb + skip
        return x





def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""
    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D
    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/utils.py
def normalize_A(config,A,lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda(config.gpu_no)-torch.eye(N,N).cuda(config.gpu_no))
    A=A+A.T


    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda(config.gpu_no)-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda(config.gpu_no)

   
    return Lnorm

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/utils.py
def generate_cheby_adj(config, L, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda(config.gpu_no))
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1],)-support[-2]
            support.append(temp)


    return support

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/layers.py
class GraphConvolution(nn.Module):

    def __init__(self, config, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda(config.gpu_no))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda(config.gpu_no))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/layers.py
class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/model.py
class Chebynet(nn.Module):
    def __init__(self, config, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.config=config
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(config, in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(self.config, L, self.K)
        
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])


        result = F.relu(result)


        return result

#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/model.py
class DGCNN(nn.Module):
    def __init__(self, config):
        #in_channels(int): The feature dimension of each electrode.
        self.config = config
        in_channels= config.in_channels
        num_electrodes=config.num_electrodes
                
        k_adj = 2
        num_classes=3
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        out_channels=256
        self.layer1 = Chebynet(config, in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, 256)
        self.fc_2 = Linear(256, config.n_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda(config.gpu_no))
        nn.init.uniform_(self.A,0.01,0.5)


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(256, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, x, alpha):
        
        x=x.reshape(x.shape[0],x.shape[1],-1)        

        x=torch.squeeze(x, axis=1)

        x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline

        L = normalize_A(self.config, self.A)
        
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)

        reverse_feature = ReverseLayerF.apply(result, alpha)
        class_output= self.fc_2(result)

        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
    

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

class MyModel(nn.Module):
    def __init__(
        self,
        config,
        input_dim=4,
        channels=32,
        num_nodes=32,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        self.device = config.device
        self.num_nodes = num_nodes
        self.output_len = 128
        diffusion_step = 1
        granularity=288

        self.Temb = TemporalEmbedding(granularity, channels)

        self.start_conv_l = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )
        self.start_conv_h = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )

        self.layers = 4

        self.DCL = DualChannelLearner(
            features = channels, 
            layers = self.layers, 
            length = self.output_len, 
            num_nodes = self.num_nodes, 
            dropout=0.1
        )

        self.mlp = nn.Sequential(
                *[MultiLayerPerceptron(channels, channels) for _ in range(3)])

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
            )
        

        self.HGL = HybridGraphLearner(
            channels=channels*2, 
            layers = self.layers, 
            num_nodes=num_nodes, 
            length = self.output_len, 
            diffusion_step=1, 
            dropout=0.1, 
            emb=None,
            device=self.device
        )

        self.fc = Linear(32*32*128, config.n_classes)

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(32*32*128, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input, alpha):
        print('input:', input.shape)
        exit()
        input = input.transpose(2,1)
        # print('i:', input.shape)
        input_data=input # input data shape: [128,32,4,128]

        residual_cpu = input_data.cpu()
        residual_numpy = residual_cpu.detach().numpy()
        coef = pywt.wavedec(residual_numpy, 'db1', level=2) #Multilevel 1D Discrete Wavelet Transform of data
        coefl = [coef[0]] + [None] * (len(coef)-1)
        coefh = [None] + coef[1:]        

        xl = pywt.waverec(coefl, 'db1')
        xh = pywt.waverec(coefh, 'db1')
        xl = torch.from_numpy(xl).to(self.device)
        xh = torch.from_numpy(xh).to(self.device)
        #start conv
        # print('x_l:', xl.shape, xh.shape)
        # exit()
        res_x = input
        x_l = self.start_conv_l(xl ) 
        x_h = self.start_conv_h(xh ) 


        #DCL
        # print('x_l:', x_l.shape, x_h.shape)
        # exit()
        dcl_out = self.DCL(x_l, x_h)
        
        x = dcl_out + self.mlp(self.start_conv(res_x ))

        x = x.view(x.shape[0], -1)
        
        class_output=self.fc(x)


        reverse_feature = ReverseLayerF.apply(x, alpha)


        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
        # return out, domain_output