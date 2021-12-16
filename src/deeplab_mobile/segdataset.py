import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Mobile_Dataset(Dataset):
    def __init__(
        self,
        images_list,
        seg_list,
        fraction=0.1,
        subset=None,
        transforms=None,
        image_color_mode="rgb",
        mask_color_mode="rgb",
    ) -> None:

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        self.transforms = transforms

        if subset not in ["Train", "Test"]:
            raise (
                ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                )
            )
        self.fraction = fraction
        self.image_list = images_list
        self.mask_list = seg_list

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int):
        image = self.image_list[index]
        mask = self.mask_list[index]

        sample = {"image": image, "mask": mask}
        sample["image"] = (
            torch.tensor(sample["image"]).expand(3, -1, -1).type(torch.ShortTensor)
        )
        sample["mask"] = (
            torch.tensor(sample["mask"]).expand(3, -1, -1).type(torch.ShortTensor)
        )
        return sample


def get_mobile_dataloaders(images, masks, batch_size=14):
    image_datasets = {
        x: Mobile_Dataset(images, masks, transforms=None, subset=x)
        for x in ["Train", "Test"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
        )
        for x in ["Train", "Test"]
    }
    return dataloaders
