import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard


def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path, weight_decay
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Use GPU if available
    if torch.cuda.is_available(): # Check if GPU is available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Move the model to the GPU
    model.to(device)
    (model.resnet).to(device)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    writer = None
    if summary_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    train_loss, n_correct = 0, 0
    count = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")
            images, labels = batch
            # Move inputs over to GPU
            images = images.to(device)
            labels = labels.to(device)

            # make prediction and compute loss
            pred = model(images)
            loss = loss_fn(pred, labels).mean()

            # update train_loss, n_correct, count
            train_loss += loss.item()
            n_correct += compute_ncorrect(torch.argmax(pred, dim=1), labels)
            count += len(labels)

            # TODO: Backpropagation and gradient descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                train_accuracy = n_correct / count
                if writer:
                    writer.add_scalar('Loss/train', train_loss, epoch)
                    writer.add_scalar('Accuracy/train', train_accuracy, epoch)

                # reset train_loss, n_correct, count
                train_loss, n_correct = 0, 0
                count = 0

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                val_loss, val_acc = evaluate(val_loader, model, loss_fn, device)
                if writer:
                    writer.add_scalar('Loss/validation', val_loss, epoch)
                    writer.add_scalar('Accuracy/validation', val_acc, epoch)
                model.train()

            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total

def compute_ncorrect(outputs, labels):
    """
    Computes the number of correct predictions that the model makes.
    """
    n_correct = (outputs == labels).sum().item()
    return n_correct

def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    model.eval()
    total_loss, n_correct = 0, 0
    count = 0
    for i, batch in enumerate(val_loader):
        # make prediction and compute loss
        images, labels = batch
        # Move inputs over to GPU
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        total_loss += loss_fn(pred, labels).mean().item()
        n_correct += compute_ncorrect(torch.argmax(pred, dim=1), labels)
        count += len(labels)

    accuracy = n_correct / count
    return total_loss, accuracy
