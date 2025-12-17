import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.functional import retrieval_average_precision

from src.dataset_retrieval import Sketchy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(opts):
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    return train_loader, val_loader

def evaluate_model(model, dataloader_test):
    with torch.no_grad():
        model.eval()
        val_step_outputs = []
        for idx, batch in enumerate(tqdm(dataloader_test)):
            sk_tensor, img_tensor, neg_tensor, category = batch[:4]
            sk_tensor, img_tensor, neg_tensor = sk_tensor.to(device), img_tensor.to(device), neg_tensor.to(device)
            img_feat = model(img_tensor, dtype='image')
            sk_feat = model(sk_tensor, dtype='sketch')
            neg_feat = model(neg_tensor, dtype='image')
            val_step_outputs.append((sk_feat, img_feat, category))
            
        Len = len(val_step_outputs)
        if Len == 0:
            return
        
        query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)])
        all_category = np.array(sum([list(val_step_outputs[i][2]) for i in range(Len)], []))
        
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            distance = -1*model.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            
            target[np.where(all_category == category)] = True
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        
        mAP = torch.mean(ap)
        return mAP.item()
    
def train_model(model, opts):
    model = model.to(device)
    dataloader_train, dataloader_test = get_dataloader(opts)
    
    loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=model.distance_fn, margin=0.2)
    
    optimizer = torch.optim.Adam([
            {'params': model.clip.parameters(), 'lr': opts.clip_LN_lr},
            {'params': [model.sk_prompt] + [model.img_prompt], 'lr': opts.prompt_lr}])
    
    mAP, avg_loss = -1e3, 0
    for i_epoch in range(opts.epochs):
        print(f"Epoch: {i_epoch+1} / {opts.epochs}")
        losses = []
        
        for _, batch in enumerate(tqdm(dataloader_train)):
            model.train()
            optimizer.zero_grad()
            
            sk_tensor, img_tensor, neg_tensor, category = batch[:4]
            sk_tensor, img_tensor, neg_tensor = sk_tensor.to(device), img_tensor.to(device), neg_tensor.to(device)
            img_feat = model(img_tensor, dtype='image')
            sk_feat = model(sk_tensor, dtype='sketch')
            neg_feat = model(neg_tensor, dtype='image')
            
            loss = loss_fn(sk_feat, img_feat, neg_feat)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
        avg_loss = sum(losses) / len(losses)
        mAP_eval = evaluate_model(model, dataloader_test)
        
        if mAP_eval >= mAP:
            mAP = mAP_eval
            torch.save(model.state_dict(), os.path.join(opts.save_dir, 'best_ckp.pth'))
            print('mAP: {:.5f}'.format(mAP_eval))
            print('Loss:{:.5f}'.format(avg_loss))