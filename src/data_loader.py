import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tensorflow.keras.utils import Sequence

# Import IMG_SIZE from model_utils (assuming it's a common constant)
from model_utils import IMG_SIZE

class DataGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs, boxes = [], []
        for _, r in batch.iterrows():
            # 1) Carga sin redimensionar, para medir tamaño real
            path = os.path.join(self.img_dir, r['Image'])
            img0 = Image.open(path).convert('RGB')
            w0, h0 = img0.size

            # 2) Ordena coordenadas y normaliza en [0,1]
            x1_raw, y1_raw = r['x_top'], r['y_top']
            x2_raw, y2_raw = r['x_bottom'], r['y_bottom']
            x1, x2 = sorted([x1_raw, x2_raw])
            y1, y2 = sorted([y1_raw, y2_raw])
            box = np.array([x1/w0, y1/h0, x2/w0, y2/h0], dtype=np.float32)

            # 3) Redimensiona imagen y escala
            img = img0.resize(IMG_SIZE)
            arr = np.array(img) / 255.0

            # 4) Augmentaciones simples
            if self.augment and np.random.rand() < 0.5:
                arr = np.fliplr(arr)
                box = np.array([1-box[2], box[1], 1-box[0], box[3]], dtype=np.float32)
            if self.augment:
                enhancer = ImageEnhance.Brightness(Image.fromarray((arr*255).astype(np.uint8)))
                arr = np.array(enhancer.enhance(np.random.uniform(0.8,1.2))) / 255.0

            imgs.append(arr)
            boxes.append(box)

        return np.stack(imgs), np.stack(boxes)