import torch as tr
from torch.utils.data import DataLoader

def test_model(model, dataset, criterion,prediction_block,prediction_criterion, batch_size=32,shuffle = False):
    # Define DataLoader for dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Test model
    model.eval()
    output_loss = 0
    prediction_loss = 0
    with tr.no_grad():
        for inputs, targets in loader:
            # Compute predictions and loss
            outputs = model(inputs)
            print(outputs.shape)
            loss = outputs.shape[0]*criterion(outputs, targets.long())
            output_loss += loss.item()

            # Compute prediction loss (could be accuracy)
            predictions = prediction_block(outputs)
            prediction_loss += len(predictions)*prediction_criterion(predictions,targets)

    # Print results
    average_output_loss = output_loss / len(dataset)
    average_prediction_loss = prediction_loss / len(dataset)
    print('Average output Loss: {:.4f}, Average prediction Loss: {:.4f}'.format(average_output_loss, average_prediction_loss))

    return average_output_loss, average_prediction_loss
