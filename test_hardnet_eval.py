"""
HarDNet-MSEG modell tesztelése az SZE adathalmazon.
Összehasonlítás a DeepLabV3+ MobileNet eredményeivel.

Használat:
    python test_hardnet_eval.py --weights external_models/HarDNet-MSEG/HarDNet-MSEG.pth

A szkript a colon_short_szd_r.ipynb-ból ismert teszt ID-kat használja,
hogy azonos tesztkészleten értékelje a modellt.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import cv2

# HarDNet-MSEG importálás
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external_models', 'HarDNet-MSEG'))
from lib.HarDMSEG import HarDMSEG

# --- Segédfüggvények (colon_short_szd_r.ipynb-ból átvéve) ---

def read_excel(excel_path):
    df = pd.read_excel(excel_path)
    if 'Wrong' in df.columns:
        df = df[df['Wrong'] != 'x']
    new_df = pd.DataFrame(columns=['ID', 'CLASS'])
    new_df['ID'] = df['ID']
    for index, row in df.iterrows():
        class_value = None
        for jnet_class in ['JNET_1', 'JNET_2A', 'JNET_2B', 'JNET_3']:
            if jnet_class in df.columns and row[jnet_class] == 'x':
                class_value = jnet_class
                break
        new_df.loc[index, 'CLASS'] = class_value
    return new_df


def read_images(excel_path):
    classes = read_excel(excel_path)
    result = pd.DataFrame(columns=['ID', 'CLASS', 'IMAGE', 'QUALITY', 'SECOND_LETTER'])
    noimagecount = 0
    images_dir = os.path.dirname(excel_path)
    
    for index, row in classes.iterrows():
        image_id = row['ID']
        image_files = [f for f in os.listdir(images_dir) if f.lower().startswith(image_id.lower())]
        if image_files:
            for image in image_files:
                img_path = os.path.join(images_dir, image)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img)
                    img_array = np.array(img_pil)
                    quality = image[6] if image[6] in ['E', 'R'] else None
                    second_letter = image[8] if len(image) > 8 else None
                    result = pd.concat([result, pd.DataFrame([{
                        'ID': image_id, 'CLASS': row['CLASS'],
                        'IMAGE': img_array, 'QUALITY': quality,
                        'SECOND_LETTER': second_letter
                    }])], ignore_index=True)
        else:
            noimagecount += 1
    print(f"Képek száma, amelyekhez nem tartozik fájl: {noimagecount}")
    return result


def prepare_sze_data(sze_df):
    """SZE adathalmaz előkészítése maszkokkal."""
    images_dir = "images"
    masks_dir = os.path.join(images_dir, "masks")
    
    prepared = []
    for _, row in sze_df.iterrows():
        image_id = row['ID']
        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().startswith(image_id.lower()) and
                       (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            mask_path = None
            for mask_suffix in ['-polyp.png', '-polyp.jpg', '_polyp.png', '_polyp.jpg']:
                candidate = os.path.join(masks_dir, base_name + mask_suffix)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            
            if mask_path is None:
                for ext in ['.png', '.jpg']:
                    candidate = os.path.join(masks_dir, base_name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
            
            if mask_path and os.path.exists(mask_path):
                prepared.append({
                    'ID': image_id,
                    'CLASS': 1,
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'image_file': img_file
                })
    
    return pd.DataFrame(prepared)


# --- HarDNet-MSEG specifikus dataset ---

class HarDNetTestDataset(data.Dataset):
    """
    Teszt adatkészlet HarDNet-MSEG modellhez.
    352x352-es képeket ad ki ImageNet normalizálással.
    """
    def __init__(self, dataframe, img_col='image_path', mask_col='mask_path'):
        self.df = dataframe
        self.img_col = img_col
        self.mask_col = mask_col
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.target_size = 352

    def __getitem__(self, index):
        img_path = self.df.iloc[index][self.img_col]
        mask_path = self.df.iloc[index][self.mask_col]
        filename = os.path.basename(img_path)
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float()
        
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).float()
        
        return img, mask, filename

    def __len__(self):
        return len(self.df)


def calculate_metrics_hardnet(outputs, targets):
    """
    HarDNet-MSEG metrikák számítása.
    Az output sigmoid-on keresztül jön (1 csatorna), nem argmax-os.
    """
    preds = (outputs > 0.5).float()
    target_binary = (targets > 0.5).float()
    
    intersection = (preds * target_binary).sum()
    pred_area = preds.sum()
    target_area = target_binary.sum()
    union = pred_area + target_area - intersection
    
    dice = (2 * intersection / (pred_area + target_area + 1e-6)).item()
    iou = (intersection / (union + 1e-6)).item()
    precision = (intersection / (pred_area + 1e-6)).item()
    recall = (intersection / (target_area + 1e-6)).item()
    
    return {'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall}


def evaluate_model(model, loader, device):
    """Modell kiértékelése a teszt adathalmazon."""
    model.eval()
    
    all_dice, all_iou, all_precision, all_recall = [], [], [], []
    
    with torch.no_grad():
        for images, masks, filenames in tqdm(loader, desc="Kiértékelés"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            
            for i in range(images.size(0)):
                metrics = calculate_metrics_hardnet(outputs[i], masks[i])
                all_dice.append(metrics['dice'])
                all_iou.append(metrics['iou'])
                all_precision.append(metrics['precision'])
                all_recall.append(metrics['recall'])
    
    results = {
        'mean_dice': np.mean(all_dice),
        'std_dice': np.std(all_dice),
        'mean_iou': np.mean(all_iou),
        'std_iou': np.std(all_iou),
        'mean_precision': np.mean(all_precision),
        'std_precision': np.std(all_precision),
        'mean_recall': np.mean(all_recall),
        'std_recall': np.std(all_recall),
        'num_samples': len(all_dice),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='HarDNet-MSEG tesztelés SZE adathalmazon')
    parser.add_argument('--weights', type=str, 
                        default='external_models/HarDNet-MSEG/HarDNet-MSEG.pth',
                        help='Betanított modell súlyok útvonala')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch méret')
    parser.add_argument('--excel_path', type=str, default='images/PI-program-JNET-classes.xlsx',
                        help='Excel fájl útvonala')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eszköz: {device}")
    
    # --- Fix teszt ID-k (colon_short_szd_r.ipynb-ból) ---
    test_ids = [
        'PI009', 'PI015', 'PI016', 'PI037', 'PI048', 'PI050', 'PI055', 'PI057', 'PI068',
        'PI072', 'PI073', 'PI078', 'PI080', 'PI085', 'PI086', 'PI091', 'PI112', 'PI113',
        'PI114', 'PI121', 'PI123', 'PI130', 'PI135', 'PI136', 'PI137', 'PI142', 'PI149',
        'PI152', 'PI166', 'PI171', 'PI175', 'PI178', 'PI181', 'PI184', 'PI187', 'PI189',
        'PI191', 'PI197', 'PI198', 'PI204', 'PI205', 'PI207', 'PI210', 'PI211', 'PI219',
        'PI224', 'PI230', 'PI238', 'PI243', 'PI247', 'PI249', 'PI252', 'PI261', 'PI266',
        'PI270', 'PI272', 'PI277', 'PI280', 'PI290', 'PI298', 'PI310', 'PI335', 'PI338',
        'PI348', 'PI349', 'PI367', 'PI368', 'PI370', 'PI372', 'PI375', 'PI376', 'PI389',
        'PI390', 'PI399', 'PI400', 'PI405', 'PI408', 'PI415', 'PI418', 'PI419', 'PI432',
        'PI433', 'PI445', 'PI454', 'PI456', 'PI462', 'PI468', 'PI472', 'PI473', 'PI478',
        'PI480',
    ]
    
    # 1. SZE adatok beolvasása
    print("SZE adatok beolvasása...")
    sze = read_images(args.excel_path)
    
    # 2. Maszkok keresése
    print("Maszkok előkészítése...")
    images_dir = "images"
    masks_dir = os.path.join(images_dir, "masks")
    
    prepared_data = []
    for _, row in sze.iterrows():
        image_id = row['ID']
        image_files = [f for f in os.listdir(images_dir) 
                       if f.lower().startswith(image_id.lower()) and 
                       (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            mask_path = None
            for mask_suffix in ['-polyp.png', '-polyp.jpg', '_polyp.png']:
                candidate = os.path.join(masks_dir, base_name + mask_suffix)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            
            if mask_path is None:
                for ext in ['.png', '.jpg']:
                    candidate = os.path.join(masks_dir, base_name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
            
            if mask_path and os.path.exists(mask_path):
                prepared_data.append({
                    'ID': image_id,
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'image_file': img_file
                })
    
    df_prepared = pd.DataFrame(prepared_data)
    print(f"Összes SZE kép maszkkal: {len(df_prepared)}")
    
    # 3. Teszt halmaz kiválogatása
    test_ids_upper = [x.upper() for x in test_ids]
    df_test = df_prepared[df_prepared['ID'].str.upper().isin(test_ids_upper)].reset_index(drop=True)
    print(f"Teszt képek száma: {len(df_test)}")
    
    if len(df_test) == 0:
        print("HIBA: Nincs teszt kép! Ellenőrizd az adatok elérési útját.")
        return
    
    # 4. HarDNet-MSEG modell betöltése
    print("\nHarDNet-MSEG modell betöltése...")
    model = HarDMSEG()
    
    if not os.path.exists(args.weights):
        print(f"HIBA: A súlyfájl nem található: {args.weights}")
        print("Töltsd le innen: https://drive.google.com/file/d/1nj-zv64RiWwYjCmWg4NME7HNf_nBncUu/view?usp=sharing")
        print(f"Mentsd ide: {args.weights}")
        return
    
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    print("Modell sikeresen betöltve!")
    
    # 5. Dataset és DataLoader
    test_dataset = HarDNetTestDataset(df_test)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=0)
    
    # 6. Kiértékelés
    print(f"\nKiértékelés {len(df_test)} teszt képen...")
    results = evaluate_model(model, test_loader, device)
    
    # 7. Eredmények
    print("\n" + "="*60)
    print("HarDNet-MSEG Eredmények (SZE Teszt Halmaz)")
    print("="*60)
    print(f"  Képek száma:    {results['num_samples']}")
    print(f"  Mean IoU:       {results['mean_iou']*100:.2f}% (±{results['std_iou']*100:.2f}%)")
    print(f"  Mean Dice:      {results['mean_dice']*100:.2f}% (±{results['std_dice']*100:.2f}%)")
    print(f"  Mean Precision: {results['mean_precision']*100:.2f}% (±{results['std_precision']*100:.2f}%)")
    print(f"  Mean Recall:    {results['mean_recall']*100:.2f}% (±{results['std_recall']*100:.2f}%)")
    print("="*60)
    
    print("\nLaTeX táblázatsor:")
    print(f"HarDNet-MSEG & {results['mean_iou']*100:.2f} & {results['mean_dice']*100:.2f} & "
          f"{results['mean_precision']*100:.2f} & {results['mean_recall']*100:.2f} \\\\")
    
    print("\nÖsszehasonlítás DeepLabV3+ MobileNet-tel:")
    print(f"  DeepLabV3+ MobileNet: Val IoU=84.49%, Test IoU=83.31%, Test Dice=90.40%")
    print(f"  HarDNet-MSEG:         Test IoU={results['mean_iou']*100:.2f}%, Test Dice={results['mean_dice']*100:.2f}%")


if __name__ == '__main__':
    main()
