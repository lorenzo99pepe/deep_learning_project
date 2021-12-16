import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.deeplab.segdataset import SegmentationDataset


def get_deeplab_dataloader(
    data_dir: str,
    image_folder: str = "Images",
    mask_folder: str = "Masks",
    fraction: float = 0.2,
    batch_size: int = 4,
):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(
            data_dir,
            image_folder=image_folder,
            mask_folder=mask_folder,
            seed=100,
            fraction=fraction,
            subset=x,
            transforms=data_transforms,
        )
        for x in ["Train", "Test"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            # shuffle=True, #to test without shuffle not done yet
            # num_workers=8
        )
        for x in ["Train", "Test"]
    }
    return dataloaders


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
