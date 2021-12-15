from matplotlib import pyplot as plt
import torch
import numpy as np

from src.utils import TYPE_NAMES


def plot_observation(images_lists, idobs):
    _, axs = plt.subplots(nrows=1, ncols=len(images_lists), figsize=(12,4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(images_lists[i][idobs]) #cmap=plt.cm.jet
        #plt.colorbar()
        plt.title('Image: {}'.format(i+1))

    plt.suptitle('Overall Title')
    plt.show()


def plot_deeplab_mobile_predictions(model, 
    images_list,
    type_to_use='flair', 
    type_names=TYPE_NAMES, 
    indexes_predict = np.arange(0, 20)
):
    for i in indexes_predict:
        input_tensor = torch.tensor(images_list[type_names.index(type_to_use)][i]).expand(3, -1, -1).type(torch.ShortTensor).float()
        truth = torch.tensor(images_list[type_names.index('seg')][i]).expand(3, -1, -1).type(torch.ShortTensor).float()

        input_batch = input_tensor.unsqueeze(0) 

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output #.argmax(0)

        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        _, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].set_title('input image')
        ax[0].axis('off')
        ax[0].imshow(input_tensor[0])
        ax[1].set_title('segmented output')
        ax[1].axis('off')
        ax[1].imshow(output_predictions[0])
        ax[2].set_title('ground truth')
        ax[2].axis('off')
        ax[2].imshow(truth[0])
        plt.show()