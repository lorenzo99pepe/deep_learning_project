import numpy as np
from PIL import Image

def load_images_from_paths(input_path, seg_path):
    used_files = sorted([f for f in input_path.iterdir()])
    seg_files = sorted([f for f in seg_path.iterdir()])

    images = []
    segs = []
    for i in range(len(used_files)):
        images.append(np.asarray(Image.open(used_files[i])))
        segs.append(np.asarray(Image.open(seg_files[i])))

    images = np.array(images)
    segs = np.array(segs)
    return images, segs