import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision

from torchmetrics.retrieval import RetrievalMAP, RetrievalPrecision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)
        self.val_step_outputs = []
        self.train_output = []
        self.best_metric = -1e3

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        self.clip.train()
        # self.sk_prompt.train()
        # self.img_prompt.train()
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        # self.backward(loss)
        optimizer.step()
        self.log('train_loss', loss)
        self.train_output.append(loss.item())
        
        return loss
    
    def on_train_epoch_end(self):
        self.train_output.clear()

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss)
        self.val_step_outputs.append({
            "sk_feat": sk_feat,
            "img_feat": img_feat,
            "category": category
        })
        

    def on_validation_epoch_end(self):
        val_step_outputs = self.val_step_outputs
        Len = len(val_step_outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([val_step_outputs[i]["sk_feat"] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i]["img_feat"] for i in range(Len)])
        all_category = np.array(sum([list(val_step_outputs[i]["category"]) for i in range(Len)], []))

        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        pr = torch.zeros(len(query_feat_all))
        top_k = 200
        
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            
            top_k_actual = min(top_k, len(gallery)) 
            top_values, top_indices = torch.topk(distance, top_k_actual, largest=True)
            
            target = torch.zeros(len(gallery), dtype=torch.bool, device=device)
            target[np.where(all_category == category)] = True
            
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=200)
            # pr[idx] = retrieval_precision(distance.cpu(), target.cpu())
            
            # target_all = torch.zeros(len(gallery), dtype=torch.bool, device=distance.device)
            # target_all[np.where(all_category == category)] = True
            # target_top_k = target_all[top_indices]
            # distance_top_k = distance[top_indices]
            # ap[idx] = retrieval_average_precision(distance_top_k.cpu(), target_top_k.cpu())
            # pr[idx] = retrieval_precision(distance_top_k.cpu(), target_top_k.cpu())
            
        mAP = torch.mean(ap)
        mpr = torch.mean(pr)
        self.log('mAP', mAP, batch_size=1)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        print ('mAP@200: {}, p@200: {}, Best mAP: {}'.format(mAP.item(), mpr.item(), self.best_metric))
        self.val_step_outputs.clear()