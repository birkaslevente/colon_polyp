import cv2

def list_cameras():
    print("Kamerák keresése (0-5 indexek)...")
    found_cameras = []
    
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.read()[0]:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Index {index}: Működik | Felbontás: {int(width)}x{int(height)} | FPS: {fps}")
            found_cameras.append(index)
            cap.release()
        else:
            print(f"Index {index}: Nem elérhető vagy foglalt")
            cap.release()
            
    return found_cameras

if __name__ == "__main__":
    found = list_cameras()
    if found:
        print(f"\nTalált működő indexek: {found}")
        print("Tipp: Általában a Laptop beépített kamerája a 0-ás.")
        print("Ha van bedugva Capture Card, az valószínűleg az 1-es vagy 2-es.")
    else:
        print("\nNem találtam kamerát.")
