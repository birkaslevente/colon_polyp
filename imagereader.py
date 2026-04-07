from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os

def read_excel(excel_path):
    df = pd.read_excel(excel_path)
    columns = df.columns
    if 'Wrong' in df.columns:
        df = df[df['Wrong'] != 'x']
    new_df = pd.DataFrame(columns=['ID', 'CLASS'])

    new_df['ID'] = df['ID']
    
    # CLASS oszlop feltöltése a JNET kategóriák alapján
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
    data = pd.DataFrame(columns=['ID', 'CLASS', 'IMAGE', 'QUALITY'])
    noimagecount = 0
    # Az images könyvtár elérési útja
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
                    
                    # Determine quality based on filename
                    quality = image[6] if image[6] in ['E', 'R'] else None
                    
                    data = pd.concat([data, pd.DataFrame([{'ID': image_id, 'CLASS': row['CLASS'], 'IMAGE': img_array, 'QUALITY': quality}])], ignore_index=True)
        else:
            print("No image found for ID: ", image_id)
            noimagecount += 1
    print("Number of images not found: ", noimagecount)
    return data

                
if __name__ == "__main__":
    excel_path = "images/PI-program-JNET-classes.xlsx"
    df = read_images(excel_path)
    print(df.head())
    print(df.shape)
