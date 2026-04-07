import sys
import os
import torch
import numpy
import PIL
import cv2
import network
from network.modeling import deeplabv3plus_mobilenet

def main():
    print("--- Environment Verification ---")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    print(f"NumPy version: {numpy.__version__}")
    print(f"Pillow version: {PIL.__version__}")
    print(f"OpenCV version: {cv2.__version__}")

    print("\n--- Model Architecture Check ---")
    try:
        # Try to instantiate a model
        # Assuming num_classes=21 for VOC as default, output_stride=16
        model = deeplabv3plus_mobilenet(num_classes=21, output_stride=16)
        print("Successfully instantiated deeplabv3plus_mobilenet")
        
        # Check for checkpoints
        ckpt_path = "checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth"
        if os.path.exists(ckpt_path):
            print(f"\nFound checkpoint: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
                print("Successfully loaded checkpoint file (cpu)")
                if "model_state" in checkpoint:
                    print("Checkpoint contains 'model_state'")
                else:
                    print("Checkpoint does not contain 'model_state' key")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"\nCheckpoint not found at {ckpt_path}, skipping load test.")

    except Exception as e:
        print(f"Error checking model: {e}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()
