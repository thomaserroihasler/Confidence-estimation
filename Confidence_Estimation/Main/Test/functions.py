import torch as tr

def test_model(model, test_loader, criterion, prediction_block, device, prediction_criterion=None, batch_size=32, shuffle=False):
    """
    Tests a model on a given test dataset using a DataLoader.

    Args:
        model: The neural network model to be tested.
        test_loader: DataLoader containing the test dataset.
        criterion: The loss function used to compute the model's output loss.
        prediction_block: A function or module to transform the model's outputs into predictions.
        device: The device to run the model on ('cuda' or 'cpu').
        prediction_criterion (optional): A criterion (like accuracy) to evaluate predictions. If None, it's not used.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data in the DataLoader.

    Returns:
        average_output_loss: The average loss of the model on the test dataset.
        average_prediction_loss (optional): The average prediction loss if prediction_criterion is provided.
    """
    model.eval()  # Prepare model for evaluation
    model.to(device)  # Ensure the model is on the correct device

    output_loss = 0
    prediction_loss = 0 if prediction_criterion is not None else None
    length = 0

    with tr.no_grad():  # Disable gradient calculations for evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the device

            outputs = model(inputs)
            loss = outputs.shape[0] * criterion(outputs, targets.long())
            output_loss += loss.item()
            length += outputs.shape[0]

            if prediction_criterion is not None:
                predictions, confidence = prediction_block(outputs)
                prediction_loss += len(predictions) * prediction_criterion(predictions, targets)

    average_output_loss = output_loss / length

    if prediction_criterion is not None:
        average_prediction_loss = prediction_loss / length
        print('Average output Loss: {:.4f}, Average prediction Loss: {:.4f}'.format(average_output_loss, average_prediction_loss))
        return average_output_loss, average_prediction_loss
    else:
        print('Average output Loss: {:.4f}'.format(average_output_loss))
        return average_output_loss
