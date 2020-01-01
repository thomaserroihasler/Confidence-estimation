# visualize images
import matplotlib.pyplot as plt

def plot_sample_images(images,labels, title, n=5):
    """Plots n sample images from the given data loader"""
    fig, axes = plt.subplots(1, n, figsize=(10, 3))

    for i, ax in enumerate(axes):
        print(images.shape)
        if images[i].shape[0] == 1:  # if grayscale
            print(images[i].shape)
            ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        else:  # if RGB
            ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')

    plt.suptitle(title)
    plt.show()


