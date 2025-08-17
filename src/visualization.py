import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .losses import sort_corners # Import sort_corners from losses module
from .data_loader import IMG_SIZE # Import IMG_SIZE from data_loader module

def plot_samples(model, df, img_dir, n=5):
    sample = df.sample(n).reset_index(drop=True)
    for _, r in sample.iterrows():
        img0 = Image.open(os.path.join(img_dir, r.Image)).convert('RGB')
        w0,h0 = img0.size
        img = img0.resize(IMG_SIZE); arr = np.array(img)/255.0
        pred = model.predict(arr[np.newaxis])[0]
        pred = sort_corners(pred).numpy()

        x1,x2 = sorted([r.x_top,r.x_bottom]); y1,y2 = sorted([r.y_top,r.y_bottom])
        gt = np.array([x1/w0,y1/h0,x2/w0,y2/h0])

        scale = np.array([IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[1]])
        x1g,y1g,x2g,y2g = gt*scale; x1p,y1p,x2p,y2p = pred*scale

        plt.figure(figsize=(4,4)); plt.imshow(img); ax=plt.gca()
        ax.add_patch(plt.Rectangle((x1g,y1g),x2g-x1g,y2g-y1g,edgecolor='blue',fill=False,lw=2))
        ax.add_patch(plt.Rectangle((x1p,y1p),x2p-x1p,y2p-y1p,edgecolor='red', fill=False,lw=2))
        inter = max(0,min(x2g,x2p)-max(x1g,x1p))*max(0,min(y2g,y2p)-max(y1g,y1p))
        union = (x2g-x1g)*(y2g-y1g)+(x2p-x1p)*(y2p-y1p)-inter+1e-6
        plt.title(f"IoU: {inter/union:.3f}"); plt.axis('off'); plt.show()