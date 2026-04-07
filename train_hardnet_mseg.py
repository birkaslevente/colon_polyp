"""
HarDNet-MSEG tanítás az SZE+Kvasir-SEG fuzionált adathalmazon.
Ugyanaz az adatpipeline és felosztás mint a DeepLabV3+ MobileNet tanításnál.

Használat:
    zenbook_venv\Scripts\python.exe train_hardnet_mseg.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import cv2
import glob as glob_module
import re

# HarDNet-MSEG importálás
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'external_models', 'HarDNet-MSEG'))
from lib.HarDMSEG import HarDMSEG

# ============================================================
# Hiperparaméterek (DeepLabV3+ tréninggel megegyező ahol lehet)
# ============================================================
TOTAL_EPOCHS = 50
EARLY_STOP_PATIENCE = 7
BATCH_SIZE = 4
LEARNING_RATE = 0.001
INPUT_SIZE = 352  # HarDNet-MSEG specifikus
NUM_WORKERS = 0
CHECKPOINT_DIR = "checkpoints_hardnet"
CHECKPOINT_PREFIX = "hardnet_mseg_best_epoch"


# ============================================================
# Structure Loss (HarDNet-MSEG eredeti loss függvénye)
# Weighted BCE + Weighted IoU
# ============================================================
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


# ============================================================
# Adatbeolvasó függvények (colon_short_szd_r.ipynb-ból átvéve)
# ============================================================
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
    result_data = []
    noimagecount = 0
    images_dir = os.path.dirname(excel_path)
    
    for _, row in classes.iterrows():
        image_id = row['ID']
        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().startswith(image_id.lower()) and
                       (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp'))]
        if image_files:
            for image in image_files:
                img_path = os.path.join(images_dir, image)
                quality = image[6] if len(image) > 6 and image[6] in ['E', 'R'] else None
                second_letter = image[8] if len(image) > 8 else None
                result_data.append({
                    'ID': image_id, 'CLASS': row['CLASS'],
                    'image_path': img_path, 'image_file': image,
                    'QUALITY': quality, 'SECOND_LETTER': second_letter
                })
        else:
            noimagecount += 1
    print(f"Hiányzó képek száma: {noimagecount}")
    return pd.DataFrame(result_data)


def find_mask(img_file, masks_dir):
    """Maszk fájl keresése az adott képhez."""
    base_name = os.path.splitext(img_file)[0]
    for mask_suffix in ['-polyp.png', '-polyp.jpg', '_polyp.png']:
        candidate = os.path.join(masks_dir, base_name + mask_suffix)
        if os.path.exists(candidate):
            return candidate
    for ext in ['.png', '.jpg']:
        candidate = os.path.join(masks_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# ============================================================
# Dataset osztály HarDNet-MSEG-hez
# ============================================================
class PolypSegDataset(data.Dataset):
    """
    Polip szegmentáció dataset HarDNet-MSEG-hez.
    - 352x352 input/maszk méret
    - ImageNet normalizálás
    - Opcionális augmentáció (flip, rotation)
    """
    def __init__(self, dataframe, img_col='image_path', mask_col='mask_path',
                 target_size=352, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_col = img_col
        self.mask_col = mask_col
        self.target_size = target_size
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __getitem__(self, index):
        img_path = self.df.iloc[index][self.img_col]
        mask_path = self.df.iloc[index][self.mask_col]
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Augmentáció (tréning)
        if self.augment:
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if np.random.random() > 0.5:
                angle = np.random.choice([90, 180, 270])
                img = img.rotate(angle)
                mask = mask.rotate(angle)
        
        # Átméretezés
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        # Numpy és normalizálás
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = torch.from_numpy(img).float()
        
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
        
        return img, mask

    def __len__(self):
        return len(self.df)


# ============================================================
# Metrikák
# ============================================================
def calculate_metrics(outputs, targets):
    """IoU és Dice metrikák 1-csatornás sigmoid outputhoz."""
    preds = (torch.sigmoid(outputs) > 0.5).float()
    target_binary = (targets > 0.5).float()
    
    intersection = (preds * target_binary).sum()
    pred_area = preds.sum()
    target_area = target_binary.sum()
    union = pred_area + target_area - intersection
    
    dice = (2 * intersection / (pred_area + target_area + 1e-6)).item()
    iou = (intersection / (union + 1e-6)).item()
    return {'dice': dice, 'iou': iou}


def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss, sample_count, dice_sum, iou_sum = 0.0, 0, 0.0, 0.0
    
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = structure_loss(outputs, masks)
        loss.backward()
        
        # Gradient clipping (az eredeti HarDNet-MSEG is használja)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        metrics = calculate_metrics(outputs.detach(), masks)
        dice_sum += metrics['dice'] * batch_size
        iou_sum += metrics['iou'] * batch_size
    
    return {
        'loss': running_loss / sample_count,
        'dice': dice_sum / sample_count,
        'iou': iou_sum / sample_count
    }


def validate(model, loader, device):
    model.eval()
    running_loss, sample_count, dice_sum, iou_sum = 0.0, 0, 0.0, 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = structure_loss(outputs, masks)
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size
            metrics = calculate_metrics(outputs, masks)
            dice_sum += metrics['dice'] * batch_size
            iou_sum += metrics['iou'] * batch_size
    
    return {
        'val_loss': running_loss / sample_count,
        'val_dice': dice_sum / sample_count,
        'val_iou': iou_sum / sample_count
    }


def evaluate_test(model, loader, device):
    """Részletes teszt kiértékelés képenként."""
    model.eval()
    all_dice, all_iou = [], []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            for i in range(images.size(0)):
                metrics = calculate_metrics(outputs[i:i+1], masks[i:i+1])
                all_dice.append(metrics['dice'])
                all_iou.append(metrics['iou'])
    
    return {
        'test_dice_mean': np.mean(all_dice), 'test_dice_std': np.std(all_dice),
        'test_iou_mean': np.mean(all_iou), 'test_iou_std': np.std(all_iou),
        'num_samples': len(all_dice)
    }


# ============================================================
# Fix ID-listák (colon_short_szd_r.ipynb-ból)
# ============================================================
val_ids = [
    'PI007', 'PI013', 'PI014', 'PI023', 'PI033', 'PI036', 'PI047', 'PI053', 'PI056',
    'PI061', 'PI065', 'PI074', 'PI077', 'PI082', 'PI084', 'PI090', 'PI097', 'PI108',
    'PI110', 'PI111', 'PI133', 'PI153', 'PI156', 'PI170', 'PI172', 'PI193', 'PI194',
    'PI196', 'PI206', 'PI209', 'PI212', 'PI213', 'PI227', 'PI229', 'PI239', 'PI254',
    'PI257', 'PI262', 'PI263', 'PI265', 'PI269', 'PI284', 'PI288', 'PI293', 'PI300',
    'PI302', 'PI305', 'PI311', 'PI319', 'PI321', 'PI325', 'PI336', 'PI347', 'PI357',
    'PI358', 'PI366', 'PI369', 'PI373', 'PI380', 'PI382', 'PI386', 'PI387', 'PI396',
    'PI398', 'PI406', 'PI412', 'PI413', 'PI416', 'PI420', 'PI431', 'PI447', 'PI448',
    'PI451', 'PI455', 'PI464', 'PI467', 'PI482', 'PI484', 'PI488',
]

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

train_ids = [
    'PI003', 'PI005', 'PI006', 'PI008', 'PI011', 'PI017', 'PI018', 'PI019', 'PI020',
    'PI021', 'PI022', 'PI024', 'PI025', 'PI026', 'PI028', 'PI029', 'PI030', 'PI031',
    'PI032', 'PI034', 'PI035', 'PI038', 'PI039', 'PI040', 'PI041', 'PI042', 'PI043',
    'PI045', 'PI046', 'PI049', 'PI051', 'PI052', 'PI058', 'PI059', 'PI060', 'PI062',
    'PI063', 'PI066', 'PI067', 'PI069', 'PI070', 'PI071', 'PI075', 'PI076', 'PI079',
    'PI081', 'PI083', 'PI087', 'PI088', 'PI089', 'PI092', 'PI093', 'PI094', 'PI095',
    'PI096', 'PI098', 'PI099', 'PI100', 'PI102', 'PI103', 'PI104', 'PI105', 'PI107',
    'PI109', 'PI115', 'PI116', 'PI118', 'PI119', 'PI120', 'PI122', 'PI124', 'PI125',
    'PI126', 'PI127', 'PI129', 'PI132', 'PI134', 'PI138', 'PI139', 'PI141', 'PI143',
    'PI144', 'PI145', 'PI146', 'PI147', 'PI148', 'PI150', 'PI151', 'PI155', 'PI157',
    'PI158', 'PI159', 'PI160', 'PI161', 'PI162', 'PI163', 'PI164', 'PI165', 'PI168',
    'PI169', 'PI174', 'PI176', 'PI177', 'PI179', 'PI180', 'PI182', 'PI185', 'PI188',
    'PI190', 'PI200', 'PI201', 'PI202', 'PI203', 'PI208', 'PI214', 'PI215', 'PI216',
    'PI217', 'PI218', 'PI221', 'PI222', 'PI225', 'PI226', 'PI228', 'PI231', 'PI232',
    'PI233', 'PI234', 'PI235', 'PI236', 'PI237', 'PI240', 'PI241', 'PI242', 'PI244',
    'PI245', 'PI246', 'PI248', 'PI250', 'PI251', 'PI253', 'PI255', 'PI256', 'PI259',
    'PI260', 'PI264', 'PI268', 'PI271', 'PI273', 'PI274', 'PI275', 'PI278', 'PI279',
    'PI281', 'PI282', 'PI283', 'PI285', 'PI291', 'PI292', 'PI294', 'PI295', 'PI296',
    'PI297', 'PI303', 'PI304', 'PI306', 'PI307', 'PI308', 'PI312', 'PI313', 'PI314',
    'PI317', 'PI318', 'PI320', 'PI322', 'PI323', 'PI327', 'PI328', 'PI330', 'PI331',
    'PI332', 'PI333', 'PI334', 'PI337', 'PI339', 'PI340', 'PI341', 'PI342', 'PI343',
    'PI344', 'PI345', 'PI350', 'PI351', 'PI352', 'PI353', 'PI354', 'PI359', 'PI360',
    'PI362', 'PI363', 'PI365', 'PI371', 'PI374', 'PI377', 'PI378', 'PI379', 'PI381',
    'PI383', 'PI384', 'PI391', 'PI392', 'PI393', 'PI394', 'PI395', 'PI397', 'PI402',
    'PI403', 'PI404', 'PI407', 'PI409', 'PI411', 'PI414', 'PI417', 'PI421', 'PI422',
    'PI423', 'PI424', 'PI425', 'PI426', 'PI427', 'PI428', 'PI429', 'PI430', 'PI434',
    'PI435', 'PI436', 'PI437', 'PI438', 'PI439', 'PI440', 'PI441', 'PI442', 'PI443',
    'PI444', 'PI446', 'PI449', 'PI452', 'PI453', 'PI457', 'PI458', 'PI459', 'PI460',
    'PI461', 'PI463', 'PI465', 'PI466', 'PI469', 'PI470', 'PI471', 'PI474', 'PI475',
    'PI477', 'PI479', 'PI481', 'PI483', 'PI485', 'PI486', 'PI487', 'PI491',
]


# ============================================================
# Fő tanító logika
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eszköz: {device}")
    
    # --- 1. SZE Adatok betöltése ---
    print("\n1. SZE adatok betöltése...")
    excel_path = "images/PI-program-JNET-classes.xlsx"
    sze_df = read_images(excel_path)
    print(f"   SZE összes kép (maszkkal): ", end="")
    
    images_dir = "images"
    masks_dir = os.path.join(images_dir, "masks")
    
    sze_with_masks = []
    for _, row in sze_df.iterrows():
        mask_path = find_mask(row['image_file'], masks_dir)
        if mask_path:
            sze_with_masks.append({
                'ID': row['ID'], 'image_path': row['image_path'],
                'mask_path': mask_path, 'image_file': row['image_file']
            })
    df_sze = pd.DataFrame(sze_with_masks)
    print(f"{len(df_sze)}")
    
    # --- 2. Kvasir-SEG betöltése ---
    print("\n2. Kvasir-SEG adatok betöltése...")
    kvasir_root = "Kvasir-SEG"
    kvasir_masks_dir = os.path.join(kvasir_root, "masks")
    
    kvasir_data = []
    if os.path.exists(kvasir_root):
        kvasir_files = [f for f in os.listdir(kvasir_root)
                        if f.endswith('.jpg') or f.endswith('.png')]
        for img_file in kvasir_files:
            img_path = os.path.join(kvasir_root, img_file)
            mask_path = os.path.join(kvasir_masks_dir, img_file)
            if not os.path.exists(mask_path):
                base_name = os.path.splitext(img_file)[0]
                for ext in ['.jpg', '.png', '.tif']:
                    alt = os.path.join(kvasir_masks_dir, base_name + ext)
                    if os.path.exists(alt):
                        mask_path = alt
                        break
            if os.path.exists(mask_path):
                kvasir_data.append({
                    'ID': 'kvasir_' + os.path.splitext(img_file)[0],
                    'image_path': img_path, 'mask_path': mask_path,
                    'image_file': img_file
                })
    df_kvasir = pd.DataFrame(kvasir_data)
    print(f"   Kvasir-SEG: {len(df_kvasir)} kép")
    
    # --- 3. Felosztás fix ID-k alapján ---
    print("\n3. Adatfelosztás fix ID-k alapján...")
    val_upper = [x.upper() for x in val_ids]
    test_upper = [x.upper() for x in test_ids]
    train_upper = [x.upper() for x in train_ids]
    
    sze_train = df_sze[df_sze['ID'].str.upper().isin(train_upper)].reset_index(drop=True)
    sze_val = df_sze[df_sze['ID'].str.upper().isin(val_upper)].reset_index(drop=True)
    sze_test = df_sze[df_sze['ID'].str.upper().isin(test_upper)].reset_index(drop=True)
    
    # Fuzionált tréning halmaz: SZE train + Kvasir-SEG
    train_df = pd.concat([sze_train, df_kvasir], ignore_index=True)
    val_df = sze_val
    test_df = sze_test
    
    print(f"   Train (SZE+Kvasir): {len(train_df)} ({len(sze_train)} SZE + {len(df_kvasir)} Kvasir)")
    print(f"   Validation (SZE):   {len(val_df)}")
    print(f"   Test (SZE):         {len(test_df)}")
    
    # --- 4. Datasetek és Loaderek ---
    print("\n4. Datasetek létrehozása...")
    train_dataset = PolypSegDataset(train_df, augment=True, target_size=INPUT_SIZE)
    val_dataset = PolypSegDataset(val_df, augment=False, target_size=INPUT_SIZE)
    test_dataset = PolypSegDataset(test_df, augment=False, target_size=INPUT_SIZE)
    
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=NUM_WORKERS)
    
    # --- 5. Modell ---
    print("\n5. HarDNet-MSEG modell inicializálása...")
    model = HarDMSEG()
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Összes paraméter: {total_params:,}")
    print(f"   Tanítható paraméter: {trainable_params:,}")
    
    # --- 6. Optimizer, Scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3
    )
    
    # --- 7. Korábbi checkpoint betöltése ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint_files = sorted(
        glob_module.glob(os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*.pth")),
        key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1))
    )
    
    start_epoch = 0
    best_iou = 0.0
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [], 'lr': []
    }
    
    if checkpoint_files:
        latest = checkpoint_files[-1]
        print(f"\n   Korábbi checkpoint betöltése: {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'iou' in ckpt:
            best_iou = ckpt['iou']
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"   Folytatás epoch {start_epoch + 1}-től, legjobb IoU: {best_iou:.4f}")
    else:
        print("\n   Nincs korábbi checkpoint, tanítás az elejéről.")
    
    # --- 8. Tanító ciklus ---
    print(f"\n{'='*60}")
    print(f"Tanítás indítása: {TOTAL_EPOCHS} epoch, batch_size={BATCH_SIZE}")
    print(f"{'='*60}")
    
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_dice'].append(val_metrics['val_dice'])
        history['val_iou'].append(val_metrics['val_iou'])
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, Dice: {val_metrics['val_dice']:.4f}, IoU: {val_metrics['val_iou']:.4f}")
        print(f"LR: {current_lr:.6f} | Best IoU: {best_iou:.4f} | No improvement: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}")
        
        scheduler.step(val_metrics['val_iou'])
        
        if val_metrics['val_iou'] > best_iou:
            best_iou = val_metrics['val_iou']
            epochs_without_improvement = 0
            save_path = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou,
                'dice': val_metrics['val_dice'],
                'history': history,
            }, save_path)
            print(f"Új legjobb modell mentve: {save_path} (IoU: {best_iou:.4f})")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: {EARLY_STOP_PATIENCE} epoch óta nincs javulás. Legjobb IoU: {best_iou:.4f}")
            break
    
    # --- 9. Teszt kiértékelés a legjobb modellel ---
    print(f"\n{'='*60}")
    print("Teszt kiértékelés a legjobb modellel...")
    print(f"{'='*60}")
    
    best_checkpoint_files = sorted(
        glob_module.glob(os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*.pth")),
        key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1))
    )
    
    if best_checkpoint_files:
        best_ckpt = torch.load(best_checkpoint_files[-1], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"Legjobb checkpoint: {best_checkpoint_files[-1]}")
        print(f"Val IoU: {best_ckpt.get('iou', 'N/A'):.4f}, Val Dice: {best_ckpt.get('dice', 'N/A'):.4f}")
    
    test_results = evaluate_test(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print("HarDNet-MSEG Teszt Eredmények (SZE Teszt Halmaz)")
    print(f"{'='*60}")
    print(f"  Képek száma:  {test_results['num_samples']}")
    print(f"  Test IoU:     {test_results['test_iou_mean']*100:.2f}% (±{test_results['test_iou_std']*100:.2f}%)")
    print(f"  Test Dice:    {test_results['test_dice_mean']*100:.2f}% (±{test_results['test_dice_std']*100:.2f}%)")
    print(f"{'='*60}")
    
    print("\nÖsszehasonlítás DeepLabV3+ MobileNet-tel:")
    print(f"  DeepLabV3+ MobileNet: Val IoU=84.49%, Test IoU=83.31%, Test Dice=90.40%")
    print(f"  HarDNet-MSEG:         Val IoU={best_iou*100:.2f}%, Test IoU={test_results['test_iou_mean']*100:.2f}%, Test Dice={test_results['test_dice_mean']*100:.2f}%")
    
    print("\nLaTeX táblázatsor:")
    print(f"HarDNet-MSEG & {best_iou*100:.2f} & {test_results['test_iou_mean']*100:.2f} & {test_results['test_dice_mean']*100:.2f} \\\\")


if __name__ == '__main__':
    main()
