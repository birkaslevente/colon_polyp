import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from live_capture_app import MedicalAnalyzer, DEVICE, INPUT_SIZE

def get_activation(name, activations_dict):
    def hook(model, input, output):
        if isinstance(output, dict):
            for k, v in output.items():
                activations_dict[f"{name}_{k}"] = v.detach()
        else:
            activations_dict[name] = output.detach()
    return hook

def save_feature_map(tensor, name, out_dir):
    """Egy tensor (aktiváció) elmentése hőtérképként és visszatérés a képpel."""
    if tensor.dim() == 4:
        feat = tensor.mean(dim=1).squeeze().cpu().numpy()
    elif tensor.dim() == 3:
        feat = tensor.squeeze().cpu().numpy()
    else:
        feat = tensor.cpu().numpy()
        
    feat = feat - np.min(feat)
    if np.max(feat) != 0:
        feat = feat / np.max(feat)
    feat_gray = np.uint8(feat * 255)
    feat_color = cv2.applyColorMap(feat_gray, cv2.COLORMAP_JET)
    
    out_path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(out_path, feat_color)
    return feat_color

def create_summary_image(images_dict, out_dir, base_name):
    """Összefoglaló kollázs készítése feliratokkal."""
    rows = []
    current_row = []
    
    # Rendezzük a kulcsokat
    sorted_keys = sorted(images_dict.keys())
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    thickness = 2
    
    cell_size = 400 # Nagyobb cellák a jobb láthatóságért
    
    for i, key in enumerate(sorted_keys):
        img = images_dict[key].copy()
        h, w = img.shape[:2]
        
        # Cellába illesztés
        cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        scale = min(cell_size/w, cell_size/h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (nw, nh))
        
        y_off = (cell_size - nh) // 2
        x_off = (cell_size - nw) // 2
        cell[y_off:y_off+nh, x_off:x_off+nw] = img_resized
        
        # Felirat a cellára
        label = f"{key} ({w}x{h})"
        cv2.putText(cell, label, (10, 30), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA) # Árnyék
        cv2.putText(cell, label, (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        current_row.append(cell)
        
        if len(current_row) == 4 or i == len(sorted_keys) - 1:
            while len(current_row) < 4:
                current_row.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))
            rows.append(np.hstack(current_row))
            current_row = []
            
    summary_img = np.vstack(rows)
    summary_path = os.path.join(out_dir, f"SUMMARY_{base_name}.png")
    cv2.imwrite(summary_path, summary_img)
    print(f"\nÖsszefoglaló kollázs mentve: {summary_path}")
    return summary_path

def main():
    print("Modell betöltése...")
    analyzer = MedicalAnalyzer(DEVICE)
    analyzer.load_models()
    
    model = analyzer.deeplab_model
    model.eval()
    
    activations = {}
    h1 = model.backbone.register_forward_hook(get_activation("1_Backbone", activations))
    h2 = model.classifier.aspp.register_forward_hook(get_activation("2_ASPP", activations))
    h3 = model.classifier.project.register_forward_hook(get_activation("3_Projected_LowLevel", activations))
    h4 = model.classifier.classifier.register_forward_hook(get_activation("4_Classifier_PreUp", activations))
    
    possible_paths = ["images/*.jpg", "images/*.png", "Kvasir-SEG/images/*.jpg", "saved_results/*_input.png", "*.jpg", "*.png"]
    import glob, random
    all_matches = []
    for p in possible_paths: all_matches.extend(glob.glob(p))
    if not all_matches: return

    test_img_path = random.choice(all_matches)
    print(f"Kiválasztott random kép: {test_img_path}")
    base_name = os.path.splitext(os.path.basename(test_img_path))[0]
    out_dir = os.path.join("feature_maps", base_name)
    os.makedirs(out_dir, exist_ok=True)

    img_bgr = cv2.imread(test_img_path)
    img_resized = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor_img, _ = analyzer.preprocess(pil_img)
    if tensor_img.dim() == 3: tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(DEVICE)
    
    with torch.no_grad():
        output = model(tensor_img)
        # DeepLabV3Plus-Pytorch modellnél az output alakja [1, 1, 513, 513] vagy [1, 2, 513, 513]
        # Ha 2 csatornás (VOC), akkor softmax kell, ha 1 csatornás, akkor sigmoid
        if output.shape[1] == 1:
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        else:
            # Polyp class a 1-es indexen van
            pred = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            
        mask_binary = np.uint8(pred > 0.5) * 255
        mask_rgb = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
        # Kényszerítsük a maszkot 513x513-ra a felirat miatt
        mask_rgb = cv2.resize(mask_rgb, (INPUT_SIZE, INPUT_SIZE))
    
    h1.remove(); h2.remove(); h3.remove(); h4.remove()
    
    # Képek gyűjtése a kollázshoz
    summary_images = {
        "0_Original": img_resized,
        "6_Binary_Mask": mask_rgb
    }
    
    print("\nAktivációk mentése...")
    for name, tensor in sorted(activations.items()):
        feat_img = save_feature_map(tensor, name, out_dir)
        summary_images[name] = feat_img
    
    # Végső kimenet (5_Final_Output) mentése külön is
    # Ha többcsatornás, vegyük a polyp csatornát
    if output.shape[1] > 1:
        final_feat_tensor = torch.softmax(output, dim=1)[:, 1:2, :, :]
    else:
        final_feat_tensor = torch.sigmoid(output)
        
    final_feat = save_feature_map(final_feat_tensor, "5_Final_Output", out_dir)
    summary_images["5_Final_Output"] = final_feat
    
    # Kollázs készítése
    create_summary_image(summary_images, out_dir, base_name)

if __name__ == "__main__":
    main()
