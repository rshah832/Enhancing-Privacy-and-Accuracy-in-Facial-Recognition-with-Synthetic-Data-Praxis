import os
import numpy as np
import PIL.Image
import torch
from IPython.display import display

import sys
sys.path.append(r'C:\...\stylegan2-ada-pytorch')

import legacy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_stylegan2_model(model_path):
    with open(model_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G
    
# Function to load a boundary
def load_boundary(file_path, device):
    boundary = np.load(file_path)
    boundary = torch.tensor(boundary, device=device, dtype=torch.float32)
    return boundary

pose_boundary = load_boundary(r'C:\...\stylegan_celebahq_pose_w_boundary.npy', device)
model_path = r'C:\...\stylegan2-celebahq-256x256.pkl'
G = load_stylegan2_model(model_path)

def generate_images_with_frontal_pose(G, num_images, pose_min_strength, pose_max_strength, seed_start, pose_boundary, output_path, noise_scale=0.1):
    os.makedirs(output_path, exist_ok=True)
    device = next(G.parameters()).device
    seeds = np.arange(seed_start, seed_start + num_images)
    pose_strengths = np.linspace(pose_min_strength, pose_max_strength, num_images)
    images = []

    for i, (seed, pose_strength) in enumerate(zip(seeds, pose_strengths)):
        np.random.seed(seed)
        z = torch.randn([1, G.z_dim], device=device)

        noise = torch.randn([1, G.z_dim], device=device) * noise_scale
        z = z + noise

        pose_strength = torch.tensor(pose_strength, device=device)
        pose_strength = torch.clamp(pose_strength, -0.1, 0.1)
        modified_z = z + pose_strength * pose_boundary

        # Generate the image
        img = G(modified_z, None)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        img = PIL.Image.fromarray(img, 'RGB')
        img.save(os.path.join(output_path, f'synthetic_image_{i}.png'))
        images.append(img)

    for img in images:
        display(img)
        
num_images = 100000
seed_start = 0
pose_min_strength = -0.1
pose_max_strength = 0.1
output_path = r'C:\...\generated_images'

generate_images_with_frontal_pose(
    G, num_images, pose_min_strength, pose_max_strength,
    seed_start, pose_boundary, output_path
)
