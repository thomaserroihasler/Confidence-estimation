import matplotlib.pyplot as plt

def plot_sample_images(images, labels, title, n=5):
    """
    Plots n sample images with their corresponding labels.

    Args:
        images (Tensor): A batch of images.
        labels (Tensor): Corresponding labels of the images.
        title (str): The title for the plot.
        n (int): Number of images to display. Default is 5.

    Note:
        The function assumes that the images are in the format (C, H, W),
        where C is the number of channels, H is the height, and W is the width.
    """
    # Ensure that the number of images to plot does not exceed the batch size
    n = min(n, len(images))

    # Create a subplot with 1 row and n columns
    fig, axes = plt.subplots(1, n, figsize=(10, 3))

    for i, ax in enumerate(axes):
        # Check if the image is grayscale (1 channel)
        if images[i].shape[0] == 1:
            # Squeeze the channel dimension and plot
            ax.imshow(images[i].squeeze().numpy(), cmap='gray')
        else:  # If RGB (3 channels)
            # Permute the dimensions from (C, H, W) to (H, W, C) for display
            ax.imshow(images[i].permute(1, 2, 0).numpy())

        # Set the title for each subplot as the label
        ax.set_title(f'Label: {labels[i].item()}')

        # Turn off axis to avoid showing ticks
        ax.axis('off')

    # Set the main title for the plot
    plt.suptitle(title)

    # Display the plot
    plt.show()
