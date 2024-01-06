import torch as tr

def test_model(model, test_loader, criterion, prediction_block, prediction_criterion=None, batch_size=32, shuffle=False):
    """
    Tests a model on a given test dataset using a DataLoader.

    Args:
        model: The neural network model to be tested.
        test_loader: DataLoader containing the test dataset.
        criterion: The loss function used to compute the model's output loss.
        prediction_block: A function or module to transform the model's outputs into predictions.
        prediction_criterion (optional): A criterion (like accuracy) to evaluate predictions. If None, it's not used.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data in the DataLoader.

    Returns:
        average_output_loss: The average loss of the model on the test dataset.
        average_prediction_loss (optional): The average prediction loss if prediction_criterion is provided.
    """
    # Prepare model for evaluation
    model.eval()
    output_loss = 0
    prediction_loss = 0 if prediction_criterion is not None else None
    length = 0
    # Disable gradient calculations for evaluation
    with tr.no_grad():
        for inputs, targets in test_loader:
            # Compute predictions and loss
            outputs = model(inputs)
            loss = outputs.shape[0] * criterion(outputs, targets.long())
            output_loss += loss.item()
            length+= outputs.shape[0]
            # Compute prediction loss (e.g., accuracy) if a secondary criterion is provided
            if prediction_criterion is not None:
                predictions, confidence = prediction_block(outputs)
                prediction_loss += len(predictions) * prediction_criterion(predictions, targets)

    # Calculate and print average losses
    average_output_loss = output_loss /(length)
    if prediction_criterion is not None:
        average_prediction_loss = prediction_loss /(length)
        print('Average output Loss: {:.4f}, Average prediction Loss: {:.4f}'.format(average_output_loss, average_prediction_loss))
        return average_output_loss, average_prediction_loss
    else:
        print('Average output Loss: {:.4f}'.format(average_output_loss))
        return average_output_loss
