from matplotlib import pyplot as plt

def plot_observation(images_lists, idobs):
    fig, axs = plt.subplots(nrows=1, ncols=len(images_lists), figsize=(12,4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(images_lists[i][idobs]) #cmap=plt.cm.jet
        #plt.colorbar()
        plt.title('Image: {}'.format(i+1))

    plt.suptitle('Overall Title')
    plt.show()