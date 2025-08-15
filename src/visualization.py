import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Import IMG_SIZE from model_utils (assuming it's a common constant)
from model_utils import IMG_SIZE, sort_corners

def plot_samples(model, df, img_dir, n=5):
    samples = df.sample(n).reset_index(drop=True)
    for _, r in samples.iterrows():
        path = os.path.join(img_dir, r['Image'])
        img0 = Image.open(path).convert('RGB')
        w0, h0 = img0.size
        img = img0.resize(IMG_SIZE)
        arr = np.array(img) / 255.0

        # predicción
        pred = model.predict(arr[np.newaxis])[0]

        # GT ordenado y normalizado
        x1r, y1r = r['x_top'], r['y_top']
        x2r, y2r = r['x_bottom'], r['y_bottom']
        x1o, x2o = sorted([x1r, x2r])
        y1o, y2o = sorted([y1r, y2r])
        gt = np.array([x1o/w0, y1o/h0, x2o/w0, y2o/h0], dtype=np.float32)

        # escala a píxeles
        scale = np.array([IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[1]])
        x1g,y1g,x2g,y2g = gt * scale
        x1p,y1p,x2p,y2p = pred * scale

        plt.figure(figsize=(5,5))
        plt.imshow(img)
        ax = plt.gca()
        ax.add_patch(patches.Rectangle((x1g,y1g),x2g-x1g,y2g-y1g,edgecolor='blue',fill=False,linewidth=2))
        ax.add_patch(patches.Rectangle((x1p,y1p),x2p-x1p,y2p-y1p,edgecolor='red', fill=False,linewidth=2))
        inter = max(0,min(x2g,x2p)-max(x1g,x1p)) * max(0,min(y2g,y2p)-max(y1g,y1p))
        union = (x2g-x1g)*(y2g-y1g)+(x2p-x1p)*(y2p-y1p)-inter+1e-6
        plt.title(f"IoU: {inter/union:.3f}")
        plt.axis('off')
        plt.show()