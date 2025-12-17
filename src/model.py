import torch
import torch.nn as nn
import torch.nn.functional as F

from src.clip import clip
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
            
class Model(nn.Module):
    def __init__(self, opts):
        super(Model, self).__init__()
        
        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=device)
        self.clip.apply(freeze_all_but_bn)
        
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        
    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat