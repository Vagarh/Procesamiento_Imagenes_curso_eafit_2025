import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tensorflow.keras.utils import Sequence
from src.config import IMG_SIZE

class DataGenerator(Sequence):
    def __init__(self, df, img_dir, batch, augment=False):
        self.df, self.dir, self.bs, self.aug = df.reset_index(drop=True), img_dir, batch, augment
    def __len__(self):
        return int(np.ceil(len(self.df)/self.bs))
    def __getitem__(self, idx):
        batch = self.df.iloc[idx*self.bs:(idx+1)*self.bs]
        imgs, boxes = [], []
        for _, r in batch.iterrows():
            img0 = Image.open(os.path.join(self.dir, r.Image)).convert('RGB')
            w0,h0 = img0.size
            x1,x2 = sorted([r.x_top, r.x_bottom]); y1,y2 = sorted([r.y_top, r.y_bottom])
            box = np.array([x1/w0, y1/h0, x2/w0, y2/h0], np.float32)

            img = img0.resize(IMG_SIZE); arr = np.array(img)/255.0
            if self.aug and np.random.rand() < .5:
                arr = np.fliplr(arr)
                box = np.array([1-box[2], box[1], 1-box[0], box[3]], np.float32)
            if self.aug:
                arr = np.array(ImageEnhance.Brightness(Image.fromarray((arr*255).astype(np.uint8)))
                               .enhance(np.random.uniform(0.8,1.2))) / 255.0
            imgs.append(arr); boxes.append(box)
        return np.stack(imgs), np.stack(boxes)