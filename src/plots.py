from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image

from src.utils import TYPE_NAMES


def plot_observation(images_lists, idobs):
    _, axs = plt.subplots(nrows=1, ncols=len(images_lists), figsize=(12, 4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(images_lists[i][idobs])  # cmap=plt.cm.jet
        # plt.colorbar()
        plt.title("Image: {}".format(i + 1))

    plt.suptitle("Overall Title")
    plt.show()


def plot_deeplab_mobile_predictions(
    model,
    images_list,
    type_to_use="flair",
    type_names=TYPE_NAMES,
    indexes_predict=np.arange(0, 20),
):
    for i in indexes_predict:
        input_tensor = torch.tensor(images_list[type_names.index(type_to_use)][i]).expand(3, -1, -1).type(torch.ShortTensor).float()
        truth = torch.tensor(images_list[type_names.index("seg")][i]).expand(3, -1, -1).type(torch.ShortTensor).float()

        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = torch.amax(output, 0).numpy()

        threshold_min = np.percentile(output_predictions, 90)
        threshold_mid = np.percentile(output_predictions, 95)
        threshold_max = np.percentile(output_predictions, 99)

        output_pred = output_predictions
        output_pred = np.where(output_predictions > threshold_min, threshold_min, 0)
        output_pred = np.where(output_predictions > threshold_mid, threshold_mid, output_pred)
        output_pred = np.where(output_predictions > threshold_max, threshold_max, output_pred)

        _, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].set_title("input image")
        ax[0].axis("off")
        ax[0].imshow(input_tensor[0])
        ax[1].set_title('segmented output')
        ax[1].axis('off')
        ax[1].imshow(output_pred)
        ax[2].set_title('ground truth')
        ax[2].axis('off')
        ax[2].imshow(truth[0])
        plt.show()


def plot_mobile_prediction_from_path(model, img_path):
    img = np.array(Image.open(img_path))
    img = img[:, :, 0]

    input_tensor = (
        torch.tensor(np.array(img)).expand(3, -1, -1).type(torch.ShortTensor).float()
    )
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = torch.amax(output, 0).numpy()

    threshold_min = np.percentile(output_predictions, 90)
    threshold_mid = np.percentile(output_predictions, 95)
    threshold_max = np.percentile(output_predictions, 99)

    output_pred = output_predictions
    output_pred = np.where(output_predictions > threshold_min, threshold_min, 0)
    output_pred = np.where(output_predictions > threshold_mid, threshold_mid, output_pred)
    output_pred = np.where(output_predictions > threshold_max, threshold_max, output_pred)

    _, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].set_title("input image")
    ax[0].axis("off")
    ax[0].imshow(input_tensor[0])
    ax[1].set_title('segmented output')
    ax[1].axis('off')
    ax[1].imshow(output_pred)
    plt.show()
