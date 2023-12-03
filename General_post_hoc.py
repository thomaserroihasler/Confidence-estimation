import os
import torch as tr
import numpy as np
from Measures import calculate_aurc, ECE_no_edge, MIE_no_edge, CrossEntropy
from Predictors import ArgmaxPredictor
import os
from torch.utils.data import  TensorDataset
import torch as tr
from Functions import ECE_no_edge, MIE_no_edge, KNNGaussianKernel,NormalizedRadialBasisKernel
from Functions import Bin_edges
from Confidences import TemperatureScaledConfidence, AveragetemperatureScaledConfidence
import torch.optim as optim
from train import train_model

Cross_entropy = False
New_prediction = False
Number_of_bins =None
Number_of_points_per_bin = None

Predictor = ArgmaxPredictor(New_prediction)

def shuffle_and_split(tensor,permutation):

    total_samples = tensor.size(0)
    # Create a permutation to shuffle the data

    # Apply the permutation to shuffle the data
    shuffled_tensor = tensor[permutation]
    # Find the midpoint
    midpoint = total_samples // 2
    # Split the tensor into two parts
    return shuffled_tensor[:midpoint], shuffled_tensor[midpoint:]


DATASET_NAME = 'CIFAR-100'  # MNIST, CIFAR-10, CIFAR-100, HAM-10000
MODEL_NAME = 'ResNet18'  # VGG, ResNet18, SimpleCNN
#TYPE_OF_EVAL = ""

base_dir = f'./Data/{MODEL_NAME.lower()}_{DATASET_NAME.lower()}.pth'

if os.path.exists(base_dir):
    print("Loading data from:", base_dir)
    tensors = tr.load(base_dir)

    # Load each tensor directly without appending to a list
    test_accuracies = tensors['test_accuracies']
    print("test_accuracies size:", test_accuracies.shape)

    test_labels = tensors['test_labels']
    print("test_labels size:", test_labels.shape)

    test_outputs = tensors['test_original_outputs']
    print("test_outputs size:", test_outputs.shape)

    test_transformed_outputs = tensors['test_transformed_outputs']
    print("test_transformed_outputs size:", test_transformed_outputs.shape)

    validation_accuracies = tensors['val_accuracies']
    print("validation_accuracies size:", validation_accuracies.shape)

    validation_labels = tensors['val_labels']
    print("validation_labels size:", validation_labels.shape)

    validation_outputs = tensors['val_original_outputs']
    print("validation_outputs size:", validation_outputs.shape)

    validation_transformed_outputs = tensors['val_transformed_outputs']
    print("validation_transformed_outputs size:", validation_transformed_outputs.shape)
else:
    print(f"Data file {base_dir} not found!")

print('validation naive output shape:', validation_outputs.shape)
print('validation transformed output shape:', validation_transformed_outputs.shape)
print('validation accuracies shape:', validation_accuracies.shape)
print('validation labels shape:', validation_labels.shape)

print('test naive output shape:', test_outputs.shape)
print('test transformed output shape:', test_transformed_outputs.shape)
print('test accuracies shape:', test_accuracies.shape)
print('test labels shape:', test_labels.shape)

# Save in Final_Data as one tensor again
final_data_dir = './Final_Data'
if not os.path.exists(final_data_dir):
    os.makedirs(final_data_dir)

final_data_path = os.path.join(final_data_dir, f'{MODEL_NAME.lower()}_{DATASET_NAME.lower()}.pth')
final_tensors = {
    'test_accuracies': test_accuracies,
    'test_labels': test_labels,
    'test_original_outputs': test_outputs,
    'test_transformed_outputs': test_transformed_outputs,
    'val_accuracies': validation_accuracies,
    'val_labels': validation_labels,
    'val_original_outputs': validation_outputs,
    'val_transformed_outputs': validation_transformed_outputs
}

# tr.save(final_tensors, final_data_path)
# print(f"Data saved in {final_data_path}")

validation_values = np.arange(0.1, 1.1, 0.1)
validation_values_list = validation_values.tolist()
validation_values_list = [1.]
diffeomorphism_number_list = [int(i) for i in range(10, 10, 1)]
DIFFEO_NUMBER_PER_SIM = 1
diffeomorphism_number_list = [DIFFEO_NUMBER_PER_SIM]
#diffeomorphism_number_list = [10,20,30]
EVAL_METHODS = {
    'Validation set size convergence': {
        'Binning Strategy': ['Number of bins'],
        'Binning Numbers': [30],

        'Validation set fraction': validation_values_list,
        'Test set fraction': [1.0],

        'Diffeomorphism Numbers': diffeomorphism_number_list,

        'Temperature scaling learning rate': [0.01],
        'Temperature scaling number of epochs': [0],
        'Temperature scaling loss': ['ECE'], #  ['Cross-entropy']
        'Temperature scaling shuffle': [False],

        'Kernel nearest neighbour number': [[50,50,50]],

    }
}

def recursive_evaluation(params, method, current_combo={}, all_results=None):
    if all_results is None:
        all_results = []

    if len(params) == 0:
        # Base case: all parameters have a value, we can evaluate
        print(f"Evaluating with {method} and parameters:")
        for key, value in current_combo.items():
            print(f"{key} = {value}")
        print("-------")
        Cross_entropy = False

        if (current_combo['Temperature scaling loss'] == 'Cross-entropy'):
            Cross_entropy = True

        New_prediction = False
        Number_of_bins = None
        Number_of_points_per_bin = None

        N_d = current_combo['Diffeomorphism Numbers']
        bin_numbers = current_combo['Binning Numbers']
        bin_strategy = current_combo['Binning Strategy']

        if (bin_strategy =='Number of bins'):
            Number_of_bins = current_combo['Binning Numbers']
        else:
            Number_of_points_per_bin = current_combo['Binning Numbers']

        ECE = ECE_no_edge(Number_of_bins, Number_of_points_per_bin)
        MIE = MIE_no_edge(Number_of_bins, Number_of_points_per_bin)

        # validation

        validation_fraction = current_combo['Validation set fraction']

        total_diffeos = validation_transformed_outputs.shape[1]
        num_evaluations = total_diffeos // N_d

        for i in range(num_evaluations):
            print(i)
            start_index = i * N_d
            end_index = start_index + N_d

            tensor_types = ['outputs', 'transformed_outputs', 'accuracies', 'labels']
            total_samples = validation_outputs.shape[0]+test_outputs.shape[0]
            permutation  = tr.randperm(total_samples)
            # Iterate over each tensor type
            for tensor_type in tensor_types:
                # Get the tensors by their variable names using the 'eval' function
                val_tensor = eval(f'validation_{tensor_type}')
                test_tensor = eval(f'test_{tensor_type}')
                concatenated_tensor = tr.cat((val_tensor, test_tensor), dim=0)
                # Shuffle and split the tensors
                split1, split2 = shuffle_and_split(concatenated_tensor, permutation)

                # Update the variables directly
                globals()[f'validation_{tensor_type}'], globals()[f'test_{tensor_type}'] = split1, split2
            v = len(validation_outputs) * validation_fraction

            v_size = int(v)  # Size of the random subset

            # Ensure we have enough data points to sample from
            if v_size > len(validation_outputs):
                raise ValueError("Not enough data points to create a subset of size int(v)")

            # Generate random indices
            indices = tr.randperm(len(validation_outputs))[:v_size]

            # Index the tensors directly
            sub_validation_naive_outputs = validation_outputs[indices]
            sub_validation_transformed_outputs = validation_transformed_outputs[indices]
            sub_validation_labels = validation_labels[indices]
            sub_validation_accuracies = validation_accuracies[indices]

            sub_validation_transformed_outputs_i = sub_validation_transformed_outputs[:,start_index:end_index]

            validation_naive_probabilities = tr.softmax(sub_validation_naive_outputs, dim=-1)
            validation_transformed_probabilities_i = tr.softmax(sub_validation_transformed_outputs_i, dim=-1)
            validation_predictions = tr.argmax(validation_naive_probabilities, dim=-1)

            validation_mean_transformed_probabilities_i = tr.mean(validation_transformed_probabilities_i[:, 0:int(N_d), :], dim=1)

            validation_mean_transformed_confidences_i = validation_mean_transformed_probabilities_i[tr.arange(validation_mean_transformed_probabilities_i.shape[0]), validation_predictions]
            validation_naive_confidences = tr.max(validation_naive_probabilities, dim=-1).values

            validation_confidence_vectors_i = tr.stack((validation_naive_confidences, validation_mean_transformed_confidences_i), dim=1)

            Temperature_scaled_naive_confidence = TemperatureScaledConfidence(1., Cross_entropy)
            Temperature_scaled_naive_optimizer = optim.SGD(Temperature_scaled_naive_confidence.parameters(),lr=current_combo['Temperature scaling learning rate'])
            validation_set_for_Temperature_scaled_naive_confidence = TensorDataset(sub_validation_naive_outputs, sub_validation_accuracies)

            Scaling_criterion = ECE_no_edge(Number_of_bins, Number_of_points_per_bin)

            if (Cross_entropy):
                Scaling_criterion = CrossEntropy(New_prediction)
                validation_set_for_Temperature_scaled_naive_confidence = TensorDataset(sub_validation_naive_outputs, sub_validation_labels)

            Temperature_scaled_naive_confidence = train_model(Temperature_scaled_naive_confidence,validation_set_for_Temperature_scaled_naive_confidence,Scaling_criterion, Temperature_scaled_naive_optimizer,
                                                              current_combo[ 'Temperature scaling number of epochs'],
                                                              len(validation_set_for_Temperature_scaled_naive_confidence),
                                                              current_combo['Temperature scaling shuffle'])

            naive_k, transformed_k, hybrid_k  = current_combo['Kernel nearest neighbour number']

            a_naive = KNNGaussianKernel(validation_naive_confidences, sub_validation_accuracies, naive_k, 0, False)
            a_transformed_i = KNNGaussianKernel(validation_mean_transformed_confidences_i, sub_validation_accuracies, transformed_k, 0, False)
            g_i = NormalizedRadialBasisKernel(validation_confidence_vectors_i, sub_validation_accuracies, hybrid_k, 0, False)
            # testing

            test_fraction = current_combo['Test set fraction']

            t = len(test_outputs) * test_fraction

            t_size = int(t)  # Size of the random subset based on test_fraction

            # Ensure we have enough data points to sample from
            if t_size > len(test_outputs):
                raise ValueError("Not enough data points to create a subset of size int(t)")

            # Generate random indices for the test subset
            test_indices = tr.randperm(len(test_outputs))[:t_size]

            # Index the tensors directly to create the test subsets
            sub_test_naive_outputs = test_outputs[test_indices]
            sub_test_transformed_outputs = test_transformed_outputs[test_indices]
            sub_test_labels = test_labels[test_indices]
            sub_test_accuracies = test_accuracies[test_indices]

            sub_test_transformed_outputs_i = sub_test_transformed_outputs[:, start_index:end_index]

            test_naive_probabilities = tr.softmax(sub_test_naive_outputs, dim=-1)
            test_transformed_probabilities_i = tr.softmax(sub_test_transformed_outputs_i, dim=-1)
            test_predictions = tr.argmax(test_naive_probabilities, dim=-1)

            test_mean_transformed_probabilities_i = tr.mean(test_transformed_probabilities_i[:, 0:int(N_d), :], dim=1)

            test_mean_transformed_confidences_i = test_mean_transformed_probabilities_i[tr.arange(test_mean_transformed_probabilities_i.shape[0]), test_predictions]
            test_naive_confidences = tr.max(test_naive_probabilities, dim=-1).values
            test_confidence_vectors = tr.stack((test_naive_confidences, test_mean_transformed_confidences_i), dim=1)

            test_temp_scaled_naive_confidences = Temperature_scaled_naive_confidence(sub_test_naive_outputs)
            print(test_naive_confidences.shape)
            test_kernel_naive_confidences = a_naive(test_naive_confidences)
            test_kernel_transformed_confidences = a_transformed_i(test_mean_transformed_confidences_i)
            test_hybrid_confidence  = g_i(test_confidence_vectors)

            # Create a dictionary to hold confidence measures and associated accuracy values
            confidence_measures = {
                'naive': test_naive_confidences,
                'temp_scaled_naive': test_temp_scaled_naive_confidences,
                'transformed': test_mean_transformed_confidences_i,
                'kernel_naive': test_kernel_naive_confidences,
                'kernel_transformed': test_kernel_transformed_confidences,
                'hybrid': test_hybrid_confidence  # Add hybrid confidence here
            }

            # Iterate over the dictionary and calculate ECE, MIE, and AURC for each confidence measure
            results = {}
            for name, confidences in confidence_measures.items():
                rcr, error, aurc = calculate_aurc(confidences,
                                                  sub_test_accuracies)  # Replace with actual AURC calculation function
                ece = ECE(confidences, sub_test_accuracies)  # Replace with actual ECE calculation function
                mie = MIE(confidences, sub_test_accuracies)  # Replace with actual MIE calculation function

                # Calculate the mean of sub_test_accuracies to get test accuracy
                test_accuracy = sub_test_accuracies.float().mean().item()  # Assuming sub_test_accuracies is a tensor

                results[name] = {
                    'ECE': ece.item(),
                    'MIE': mie.item(),
                    'AURC': aurc.item(),
                    'Test Accuracy': test_accuracy  # Adding test accuracy here
                }
                print(name + ' ', results[name])

            # Append this evaluation's results to all_results
            all_results.append({
                'method': method,
                'parameters': current_combo.copy(),
                'results': results
            })

        return all_results

        # Pick one parameter and loop over its values
    param = list(params.keys())[0]
    values = params[param]

    for value in values:
        # Recur with one less parameter, and updated current_combo and all_results
        new_params = {k: v for k, v in params.items() if k != param}
        new_combo = {**current_combo, param: value}
        recursive_evaluation(new_params, method, new_combo, all_results)

    # After the last recursive call, you can choose to return all_results
    # It will have collected all the results from each recursive call
    return all_results

# Usage
for method, params in EVAL_METHODS.items():
    all_evaluations = recursive_evaluation(params, method)
    #print('hello')
    # After all evaluations, save all_results to Final_Data directory
    res_and_eval_dir = './Final_Data/'+f'{MODEL_NAME.lower()}_{DATASET_NAME.lower()}'
    if not os.path.exists(res_and_eval_dir):
        os.makedirs(res_and_eval_dir)
    results_path = os.path.join(final_data_dir, f'{MODEL_NAME.lower()}_{DATASET_NAME.lower()}/results.pth')
    eval_method_path = os.path.join(final_data_dir, f'{MODEL_NAME.lower()}_{DATASET_NAME.lower()}/eval_method.pth')
    print('final result path',results_path)
    tr.save({'all_evaluations': all_evaluations}, results_path)
    tr.save({'Method':method,'Parameters':params} ,eval_method_path)
    print(f"Evaluation results and method saved in {results_path} and {eval_method_path}")
