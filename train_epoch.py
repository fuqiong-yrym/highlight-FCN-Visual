import torch
from torch import nn
from utils.helpers import to_device
from utils.validation import IoULoss
from utils.test import test_dataset_accuracy

def train_model(model, train_loader, test_loader, optimizer, epoch, writer):
    
    to_device(model.train())
    cel = False
    if cel:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = IoULoss(softmax=True)
    # end if

    running_loss = 0.0
    running_samples = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)

        # The ground truth labels have a channel dimension (NCHW).
        # We need to remove it before passing it into
        # CrossEntropyLoss so that it has shape (NHW) and each element
        # is a value representing the class of the pixel.
        if cel:
            targets = targets.squeeze(dim=1)
        
        loss = criterion(outputs['out'], targets)
        loss.backward()
        optimizer.step()

        running_samples += targets.size(0)
        running_loss += loss.item()
    
    writer.add_scalar('training loss', running_loss/(batch_idx+1), epoch)
    with torch.inference_mode():
        iou, pixel_accuracy, custom_iou = test_dataset_accuracy(model, test_loader)
        writer.add_scalar('iou accuracy', iou, epoch)
        writer.add_scalar('pixel accuracy', pixel_accuracy, epoch)
        writer.add_scalar('custom iou', custom_iou, epoch)
    print("Trained {} samples, Loss: {:.4f}".format(
        running_samples,
        running_loss / (batch_idx+1),
    ))
