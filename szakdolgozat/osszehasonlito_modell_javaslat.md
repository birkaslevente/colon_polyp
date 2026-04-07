# Összehasonlító modell javaslat: U-Net (segmentation_models_pytorch)

## Miért ez a modell?

- **Biztosan be lehet üzemelni:** Nincs szükség külön polip-súlyokra. Az encoder (pl. ResNet34) ImageNet előtanított súlyai a könyvtárból automatikusan letölthetők.
- **Ugyanaz az adat:** Ugyanaz a fuzionált adathalmaz (SZE + Kvasir-SEG), ugyanaz a train/val/test felosztás (fix ID-k), így tisztán összehasonlítható a DeepLabV3+-mal.
- **Egyszerű integráció:** Egy sorban lehet létrehozni a modellt; a bemenet/kimenet (kép → bináris maszk) kompatibilis a meglévő pipeline-dal.
- **Irodalom:** U-Net polip szegmentálásra széles körben használt baseline (pl. Kvasir-SEG benchmarkok).

## Technikai rövid

| | DeepLabV3+ MobileNet | U-Net (smp, ResNet34) |
|--|----------------------|------------------------|
| Encoder | MobileNetV2 | ResNet34 (ImageNet) |
| Kimenet | 2 osztály (logit) | 1 csatorna (sigmoid) |
| Bemenet | 513×513 (vagy 352) | 256×256 vagy 512×512 |
| Súlyok | Polip/VOC finetune + saját tanítás | Csak ImageNet encoder + saját tanítás |

## Lépések

1. Függőség: `pip install segmentation-models-pytorch`
2. Tanítás: ugyanazzal a train/val/test DataFrame-fel és fix ID-kkal, pl. `train_unet_smp.py` (létrehozva).
3. Kiértékelés: ugyanazon teszt halmazon IoU és Dice, hogy táblázatba tudd tenni a DeepLab mellett.

## Rövid kódpélda (modell)

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
# Loss: BCEWithLogitsLoss vagy Dice. Ugyanaz az adatbetöltés, metrikák.
```
