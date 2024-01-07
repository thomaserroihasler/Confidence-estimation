import matplotlib.pyplot as plt
import torch as tr
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


def plot_images_from_loader(loader, title, n=5):
    # Get the first batch of images and labels
    images, labels = next(iter(loader))

    # In case of transformed dataset, reshape the images
    if images.dim() == 5:  # Assuming shape [batch_size, N, channels, height, width]
        batch_size, N, C, H, W = images.size()
        images = images.view(-1, C, H, W)  # Merge batch and N dimensions
        labels = labels.repeat_interleave(N)  # Repeat labels N times for each transformed image

    # Plot the images
    plot_sample_images(images, labels, title, n=n)

# reliability diagram

def reliability_diagram(bin_edges, confidences, accuracies,title = 'Reliability Diagram', save_path = './' ,legend=False):
    """
    Computes a reliability diagram (as a histogram) given the edges of the confidence bins,
    the confidences, and the accuracies.

    Args:
    - bin_edges (Tensor): A tensor of bin edges defining the bins for the reliability diagram.
    - confidences (Tensor): A tensor of confidence scores for each prediction.
    - accuracies (Tensor): A tensor of binary accuracies indicating whether each prediction is correct.
    - save_path (str): The path where to save the reliability diagram figure.

    Returns:
    - hist (Tensor): A tensor of counts representing the reliability diagram.
    - bin_edges (Tensor): A tensor of bin edges for the reliability diagram.
    """

    with tr.no_grad():
        # Compute the bin counts and accuracies
        bin_counts = tr.zeros(len(bin_edges) - 1)
        bin_accuracies = tr.zeros(len(bin_edges) - 1)
        print(bin_counts)
        for i in range(len(bin_edges) - 1):
            if (i == 0):
                mask = ((confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1]))
            else:
                mask = ((confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1]))
            if tr.sum(mask) > 0:
                bin_counts[i] = tr.sum(mask)
                bin_accuracies[i] = tr.sum(accuracies[mask]) / tr.sum(mask)
        normalized_bin_counts = tr.log(bin_counts+1)/(tr.max(tr.log(bin_counts+1)).item())
        #print('the bin counts and normalized bin counts are ', bin_counts, normalized_bin_counts)
        # Compute the bar plot
        print(bin_accuracies)
        fig, ax = plt.subplots()
        for i in range(len(bin_edges) - 1):
            if bin_counts[i] > 0:
                height = bin_accuracies[i].item()
                ax.bar(bin_edges[i], height, width=tr.diff(bin_edges)[i], alpha=normalized_bin_counts[i].item(), align='edge', color='blue')
                ax.plot([bin_edges[i], bin_edges[i + 1]], [height, height], 'k-', linewidth=2,alpha=1)

        # Plot the perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # Set the x and y axis labels and title
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)

        if legend:
            ax.legend()

        # Save the figure to file
        plt.show()
        #fig.savefig(save_path)
        #fig.clear()
        bin_confidences = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_counts,bin_accuracies, bin_confidences

def scatter_plot_tensors(tensor1, tensor2, save_path):
    assert tensor1.shape == tensor2.shape, "Both tensors must be of the same shape"
    plt.scatter(tensor1, tensor2)
    plt.savefig(save_path)
    plt.close()
