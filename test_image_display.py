#!/usr/bin/env python3
"""
Kép betöltési és megjelenítési teszt script
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from utils import ext_transforms as et

def test_image_loading():
    """Teszteli a kép betöltést és megjelenítést"""
    
    # 1. Közvetlen kép betöltés PIL-lel
    image_path = 'images/pi003-E-B.jpg'
    
    if not os.path.exists(image_path):
        print(f"Kép nem található: {image_path}")
        return
    
    print(f"Kép betöltése: {image_path}")
    
    # PIL betöltés
    img_pil = Image.open(image_path).convert('RGB')
    img_array = np.array(img_pil)
    
    print(f"PIL kép mérete: {img_array.shape}")
    print(f"PIL pixel értékek: {img_array.min()} - {img_array.max()}")
    print(f"PIL átlag RGB: {img_array.mean(axis=(0,1))}")
    
    # 2. Transzformációk tesztelése
    transform = et.ExtCompose([
        et.ExtResize(size=513),
        et.ExtCenterCrop(size=513),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dummy maszk létrehozása
    mask_pil = Image.new('L', img_pil.size, 0)
    
    # Transzformáció alkalmazása
    img_tensor, mask_tensor = transform(img_pil, mask_pil)
    
    print(f"Tensor kép mérete: {img_tensor.shape}")
    print(f"Tensor értékek: {img_tensor.min()} - {img_tensor.max()}")
    print(f"Tensor átlag csatornánként: {img_tensor.mean(dim=(1,2))}")
    
    # 3. Helyes denormalizálás
    def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """Helyes denormalizálás"""
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        
        # Denormalizálás: img = img * std + mean
        denorm = tensor * std + mean
        denorm = torch.clamp(denorm, 0, 1)  # 0-1 tartományba vágás
        
        return denorm
    
    # 4. Hibás denormalizálás (mint a notebook-ban)
    def wrong_denormalize(tensor):
        """Hibás denormalizálás (mint a notebook-ban)"""
        img_np = tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        return img_np
    
    # Denormalizálás tesztelése
    img_denorm_correct = denormalize_image(img_tensor)
    img_denorm_wrong = wrong_denormalize(img_tensor.clone())
    
    # Megjelenítés
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Eredeti kép
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Eredeti kép (PIL)')
    axes[0, 0].axis('off')
    
    # Normalizált kép (0-1)
    axes[0, 1].imshow(img_array / 255.0)
    axes[0, 1].set_title('Eredeti kép (0-1 normalizált)')
    axes[0, 1].axis('off')
    
    # Tensor vizualizáció (nyers)
    tensor_vis = img_tensor.permute(1, 2, 0).numpy()
    axes[0, 2].imshow(tensor_vis)
    axes[0, 2].set_title('Tensor (nyers, normalizált)')
    axes[0, 2].axis('off')
    
    # Helyes denormalizálás
    correct_vis = img_denorm_correct.permute(1, 2, 0).numpy()
    axes[1, 0].imshow(correct_vis)
    axes[1, 0].set_title('Helyes denormalizálás')
    axes[1, 0].axis('off')
    
    # Hibás denormalizálás
    axes[1, 1].imshow(img_denorm_wrong)
    axes[1, 1].set_title('Hibás denormalizálás (notebook)')
    axes[1, 1].axis('off')
    
    # Min-max normalizálás az eredeti tensor-on
    tensor_minmax = tensor_vis.copy()
    tensor_minmax = (tensor_minmax - tensor_minmax.min()) / (tensor_minmax.max() - tensor_minmax.min() + 1e-8)
    axes[1, 2].imshow(tensor_minmax)
    axes[1, 2].set_title('Min-max normalizálás')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_display_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nEredmények:")
    print(f"Helyes denormalizálás értékek: {correct_vis.min():.3f} - {correct_vis.max():.3f}")
    print(f"Hibás denormalizálás értékek: {img_denorm_wrong.min():.3f} - {img_denorm_wrong.max():.3f}")
    print(f"Min-max normalizálás értékek: {tensor_minmax.min():.3f} - {tensor_minmax.max():.3f}")

if __name__ == "__main__":
    test_image_loading() 