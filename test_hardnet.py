import sys
sys.path.insert(0, 'external_models/HarDNet-MSEG')

import torch
from lib.HarDMSEG import HarDMSEG

print("Modell példányosítás...")
model = HarDMSEG()
model.eval()

print("Teszt input generálás (1, 3, 352, 352)...")
x = torch.randn(1, 3, 352, 352)

with torch.no_grad():
    out = model(x)

print(f"OK! Output shape: {out.shape}")
print(f"Output min: {out.min().item():.4f}, max: {out.max().item():.4f}")
print("A modell sikeresen fut a jelenlegi környezetben!")
