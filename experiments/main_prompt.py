import torch
import argparse

from src.model import Model
from src.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sketch VLM CVPR2023')
    parser.add_argument('--exp_name', type=str, default='LN_prompt')
    
    # DataLoader Options
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/sketchy/Sketchy') 
    parser.add_argument('--save_dir', type=str, default='') 
    parser.add_argument('--max_size', type=int, default=224)
    parser.add_argument('--nclass', type=int, default=10)
    parser.add_argument('--data_split', type=float, default=-1.0)
    
    # Training Params
    parser.add_argument('--clip_lr', type=float, default=1e-4)
    parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
    parser.add_argument('--prompt_lr', type=float, default=1e-5)
    parser.add_argument('--linear_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    
    # ViT Prompt Parameters
    parser.add_argument('--prompt_dim', type=int, default=768)
    parser.add_argument('--n_prompts', type=int, default=3)
    
    opts = parser.parse_args()
    model = Model(opts).to(device)
    train_model(model, opts=opts)