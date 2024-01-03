import torch as tr
import numpy as np
from Confidence_Estimation.Configurations.Configurations import*
device = tr.device('cuda:0' if tr.cuda.is_available() else 'cpu')

def load_and_preprocess_data(dataset_name, classes_to_include=None):
    transform = transforms.Compose(CONFIG[dataset_name]['transforms'])
    test_and_val_size = 0
    if dataset_name != 'HAM-10000':
        train_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=True, download=True,transform=transform)
        test_dataset = CONFIG[dataset_name]['loader'](root=CONFIG[dataset_name]['path'], train=False, download=True, transform=transform)
        test_dataset_size = len(test_dataset)

    else:
        # For HAM-10000 custom dataset
        full_dataset = CONFIG[dataset_name]['loader'](CONFIG[dataset_name]['path'], CONFIG[dataset_name]['label_path'], transform=transform)
        total_size = len(full_dataset)
        train_size = total_size - int(0.3 * total_size)  # Remaining 70% for training
        test_dataset_size = total_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_dataset_size])

    # If there are specific classes to include, filter datasets
    if classes_to_include:
        train_dataset = filter_classes(train_dataset, classes_to_include, dataset_name)
        test_dataset = filter_classes(test_dataset, classes_to_include, dataset_name)
        test_dataset_size = len(test_dataset)
        val_size = test_dataset_size // 2
        test_size = test_dataset_size - val_size

    val_size = test_dataset_size // 2
    test_size = test_dataset_size - val_size
    print("val_size:", val_size)
    print("test_size:", test_size)
    print("test_dataset_size (after split):", len(test_dataset))
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, device = device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = tr.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
