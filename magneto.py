import torch.nn as nn
import math
from timm.models.layers import DropPath, to_2tuple

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.ln1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.ln2 = nn.LayerNorm(hidden_features)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.ln2(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def _init_weights(self,gamma):
        nn.init.xavier_normal_(self.fc1.weight, gain=gamma)
        nn.init.xavier_normal_(self.fc2.weight, gain=gamma)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.ln1 = nn.LayerNorm(dim)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.ln2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask = None):
        B, N, C = x.shape

        x = self.ln1(x)
        qk = self.qk(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask, float('-inf')) if mask is not None else attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.ln2(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _init_weights(self,gamma):
        nn.init.xavier_normal_(self.v.weight,gain=gamma)
        nn.init.xavier_normal_(self.qk.weight,gain=1)
        nn.init.xavier_normal_(self.proj.weight,gain=gamma)

class CrossAttention(nn.Module):
    def __init__(self, x_dim, y_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert x_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = x_dim // num_heads
        self.scale = head_dim ** -0.5

        self.ln = nn.LayerNorm(x_dim)
        self.q = nn.Linear(x_dim, x_dim, bias=qkv_bias)
        self.kv = nn.Linear(y_dim, x_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(x_dim, x_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.xavier_normal_(self.q.weight,gain=1)
        nn.init.xavier_normal_(self.kv.weight,gain=1)
        nn.init.xavier_normal_(self.proj.weight,gain=1)

    def forward(self, x, y):
        B, N, C = x.shape
        x = self.ln(x)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_normal_(self.q.weight,gain=1)
        nn.init.xavier_normal_(self.kv.weight,gain=1)
        nn.init.xavier_normal_(self.proj.weight,gain=1)

class Encdoer(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=False,drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU):
        super().__init__()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask = None):
        x = x + self.drop_path1(self.attn(x, mask))
        x = x + self.drop_path2(self.mlp(x))
        return x

    def _init_weights(self,gamma):
        self.attn._init_weights(gamma)
        self.mlp._init_weights(gamma)

class Decoder(nn.Module):
    def __init__(self,x_dim,y_dim,num_heads,mlp_ratio=4.,qkv_bias=False,drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU):
        super().__init__()
        self.attn1 = Attention(x_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn2 = CrossAttention(x_dim, y_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=x_dim, hidden_features=int(x_dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, y, mask=None):
        x = x + self.drop_path1(self.attn1(x, mask))
        x = x + self.drop_path2(self.attn2(x, y))
        x = x + self.drop_path3(self.mlp(x))
        return x

    def _init_weights(self,gamma):
        self.attn1._init_weights(gamma)
        self.attn2._init_weights(gamma)
        self.mlp._init_weights(gamma)

def get_gamma(encoder_layer=0,decoder_layer=0):
    N,M = encoder_layer,decoder_layer
    gamma1,gamma2 = None,None
    if M > 0 and N > 0:
        gamma1 = math.sqrt(1/3 * math.log(3*M) * math.log(2*N))
        gamma2 = math.sqrt(math.log(3*M))
    elif M > 0:
        gamma2 = math.sqrt(math.log(2*M))
    elif N > 0:
        gamma1 = math.sqrt(math.log(2*N))
    return gamma1,gamma2

def test():
    a,b = get_gamma(5,5)
    model_a = Encdoer(512,8)
    model_b = Decoder(512,512,8)
    model_a._init_weights(a)
    model_b._init_weights(b)

test()
