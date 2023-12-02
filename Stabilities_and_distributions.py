import torch as tr
import torch.nn as nn

from DataProcessing import generate_noises_on_sphere
from Figures import plot_tensor
#### Different stabilities to a set of transformations


### Relative stability of a network arround x between a set of N transformations and N noises of norm equal to the average norm of the transformation

# Generalized noise relative stability

def noise_relative_stability(transformations, network, x_batch):

    batch_size = x_batch.size(0)
    transformed_inputs = tr.stack([transf(x_batch) for transf in transformations])
    input_shape = x_batch.shape[1:]
    num_transformations = len(transformations)

    outputs_x = network(x_batch)

    #print('the output shape is', outputs_x.shape)

    transformed_outputs = network(transformed_inputs.view(-1, *x_batch.shape[1:])).view(num_transformations, batch_size, -1)

    #print('the transformed output shape is', transformed_outputs.shape)

    #print('the tensor in the norm is',(outputs_x.unsqueeze(0),transformed_outputs,(outputs_x.unsqueeze(0)- transformed_outputs)))

    trsf_norms_squared = tr.norm(outputs_x.unsqueeze(0) - transformed_outputs, dim=-1, p=2) ** 2

    #print('the trsf_norms_squared in the norm is',trsf_norms_squared.shape)

    T = tr.median(trsf_norms_squared, dim = 0).values

    diffs = transformed_inputs - x_batch.unsqueeze(0)

    #print('the diffs in the norm is', diffs.shape)

    new_shape = tuple(diffs.shape[:-3])+(-1,)
    diffs = diffs.view(new_shape)
    norms = tr.norm(diffs, p='fro', dim=-1)
    #print('norms',norms)
    #print('the norms is', norms.shape)

    r = tr.mean(norms, dim = 0)

    #print('the input shape is',x_batch.shape[1:])

    #print('the number of transformations is ',num_transformations)

    #print('the radii are ',r.shape)
    noises = generate_noises_on_sphere(r,num_transformations,x_batch.shape[1:])

    noise_inputs = x_batch.unsqueeze(1) + noises
    old_shape = tuple(noise_inputs.shape[0:2])
    new_shape = (-1,) + tuple(noise_inputs.shape[2:])
    noise_inputs = noise_inputs.view(new_shape)
    #print('the noise input shape is', noise_inputs.shape)
    noise_outputs = network(noise_inputs).view(old_shape+(-1,))
    noise_diff_norms_squared = tr.norm(outputs_x.unsqueeze(1) - noise_outputs, dim=2, p=2) ** 2
    N = tr.median(noise_diff_norms_squared, dim = 1).values

    #print('the fraction matrix is', T/N)
    return T / N

### Coherency of how the network classifies the transformed inputs the same way as the original

def coherency(transformations, network, x_batch):
    batch_size = x_batch.size(0)
    transformed_inputs = tr.stack([transf(x_batch) for transf in transformations])
    # print('the input is',x_batch)
    # print('the transformed input are', transformed_inputs)
    num_transformations = len(transformations)
    #print(num_transformations)
    outputs_x = network(x_batch)

    transformed_outputs = network(transformed_inputs.view(-1, *x_batch.shape[1:])).view(num_transformations, batch_size, -1)

    # Get the predicted class for each input in the batch
    _, predicted_x = tr.max(outputs_x, dim=-1)

    # Get the predicted class for each transformed input in the batch
    _, predicted_transformed = tr.max(transformed_outputs, dim=-1)
    # Count the number of transformed inputs that have the same predicted class as the original input
    #print(predicted_transformed,predicted_x.repeat(num_transformations, 1))
    coherency_scores = tr.mean((predicted_transformed == predicted_x.repeat(num_transformations, 1)).float(), dim=0)
    #print(coherency_scores)
    # Return the coherency score for each input in the batch
    print('coherency scores',coherency_scores)
    return coherency_scores


def coherency_probability(transformations, network, x_batch):
    batch_size = x_batch.size(0)
    transformed_inputs = tr.stack([transf(x_batch) for transf in transformations])
    num_transformations = len(transformations)
    # Plot the images
    #plot_tensor(transformed_inputs.view(-1, *x_batch.shape[1:]))
    #print(transformed_inputs.view(-1, *x_batch.shape[1:]).shape)
    outputs_x = network(x_batch)
    #print('the outputs',outputs_x)
    transformed_outputs = network(transformed_inputs.view(-1, *x_batch.shape[1:])).view(num_transformations, batch_size,                                                                                    -1)

    # Apply softmax to get probabilities
    softmax = nn.Softmax(dim=-1)
    prob_x = softmax(outputs_x)
    prob_transformed = softmax(transformed_outputs)
    #print('original and transformed probabilities',prob_x,prob_transformed)
    # Get the predicted class for each input in the batch
    _, predicted_x = tr.max(outputs_x, dim=-1)
    #print('the prediction',predicted_x)
    # Collect the probabilities of the predicted class for each transformed input
    predicted_transformed_probs = prob_transformed[:, tr.arange(batch_size), predicted_x]
    #print('transformed probabilities',predicted_transformed_probs)
    # Compute the mean probability for each input in the batch
    coherency_scores = tr.mean(predicted_transformed_probs, dim=0)
    #print('probability coherency',coherency_scores)
    # Return the coherency score for each input in the batch

    return coherency_scores


#
#
# def coherency(transformations, network, x):
#     # Apply each transformation to the input x
#     transformed_inputs = tr.stack([transf(x) for transf in transformations])
#
#     # Compute the number of transformations
#     N = len(transformations)
#
#     # Get the network's output for the original input x
#     outputs_x = network(x.unsqueeze(0)).squeeze()
#
#     # Get the network's output for the transformed inputs
#     outputs = network(transformed_inputs)
#
#     # Get the predicted class for the original input x
#     predicted_x = tr.argmax(outputs_x)
#
#     # Get the predicted class for each transformed input
#     predicted_transformed = tr.argmax(outputs, dim=-1)
#
#     # Compute the proportion of transformed inputs that have the same predicted class as the original input
#     return tr.mean((predicted_transformed == predicted_x).float())

## Absolute stabilities

class Absolute_Stability(nn.Module):
    def __init__(self, transforms):
        super(Absolute_Stability, self).__init__()
        self.transforms = transforms

    def forward(self, x):
        return self.stability(x, self.transforms)

    def stability(self, x, transforms):
        raise NotImplementedError('Subclasses of Stability must implement the stability method.')

## Relative stability

class RelativeStability(nn.Module):
    def __init__(self, transforms1, transforms2, stability_fn):
        super(RelativeStability, self).__init__()
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.stability_fn = stability_fn

    def forward(self, x):
        return self.stability_fn(x, self.transforms1, self.transforms2)