from matplotlib import pyplot as plt
from utils.helpers import *
import torch
import torchvision
from torch import nn
from utils.validation import *
import torchmetrics as TM
import torchvision.transforms as T
import os

t2img = T.ToPILImage()
img2t = T.ToTensor()

def prediction_accuracy(ground_truth_labels, predicted_labels):
    eq = ground_truth_labels == predicted_labels
    return eq.sum().item() / predicted_labels.numel()
    
def print_test_dataset_masks(model, test_pets_targets, test_pets_labels, save_path, show_plot):
    to_device(model.eval())
    predictions = model(to_device(test_pets_targets))
    predictions = predictions['out']
    test_pets_labels = to_device(test_pets_labels)
    # print("Predictions Shape: {}".format(predictions.shape))
    pred = nn.Softmax(dim=1)(predictions)

    pred_labels = pred.argmax(dim=1) #(B, H, W)
    # Add a value 1 dimension at dim=1
    pred_labels = pred_labels.unsqueeze(1) #(B, 1, H, W)
    # print("pred_labels.shape: {}".format(pred_labels.shape))
    pred_mask = pred_labels.to(torch.float) #(B, 1, H, W)
    
    # accuracy = prediction_accuracy(test_pets_labels, pred_labels)
    # iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND))
    iou = to_device(TM.classification.BinaryJaccardIndex())
    iou_accuracy = iou(pred_mask, test_pets_labels)
    #pixel_metric = to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
    pixel_metric = to_device(TM.classification.BinaryAccuracy())
    pixel_accuracy = pixel_metric(pred_labels, test_pets_labels)
    custom_iou = IoUMetric(pred, test_pets_labels)
    #title = f'Epoch: {epoch:02d}, Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom IoU: {custom_iou:.4f}]'
    #print(title)
    # print(f"Accuracy: {accuracy:.4f}")

    # Close all previously open figures.
    close_figures()
    
    fig = plt.figure(figsize=(10, 12))
    #fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    #plt.imshow(t2img(torchvision.utils.make_grid(test_pets_targets, nrow=7)))
    grid = torchvision.utils.make_grid(test_pets_targets, nrow=7)
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    #plt.imshow(t2img(torchvision.utils.make_grid(test_pets_labels.float() / 2.0, nrow=7)))
    # int64 (torch.long) change to float
    grid = torchvision.utils.make_grid(test_pets_labels.float(), nrow=7)
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    #plt.imshow(t2img(torchvision.utils.make_grid(pred_mask / 2.0, nrow=7)))
    grid = torchvision.utils.make_grid(pred_mask, nrow=7)
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title("Predicted Labels")
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "one_test_batch.png"), format="png", bbox_inches="tight", pad_inches=0.4)
    # end if
    
    if show_plot is False:
        close_figures()
    else:
        plt.show()


def test_dataset_accuracy(model, loader):
    to_device(model.eval())
    #iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro')
    #pixel_metric = to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
    iou = to_device(TM.classification.BinaryJaccardIndex())
    pixel_metric = to_device(TM.classification.BinaryAccuracy())
                    
    iou_accuracies = []
    pixel_accuracies = []
    custom_iou_accuracies = []
    
    print_model_parameters(model)

    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        inputs = to_device(inputs)
        targets = to_device(targets)
        predictions = model(inputs)
        predictions = predictions['out']
        pred_probabilities = nn.Softmax(dim=1)(predictions)
        pred_labels = predictions.argmax(dim=1)

        # Add a value 1 dimension at dim=1
        pred_labels = pred_labels.unsqueeze(1)
        # print("pred_labels.shape: {}".format(pred_labels.shape))
        pred_mask = pred_labels.to(torch.float)

        iou_accuracy = iou(pred_mask, targets)
        # pixel_accuracy = pixel_metric(pred_mask, targets)
        pixel_accuracy = pixel_metric(pred_labels, targets)
        custom_iou = IoUMetric(pred_probabilities, targets)
        iou_accuracies.append(iou_accuracy.item())
        pixel_accuracies.append(pixel_accuracy.item())
        custom_iou_accuracies.append(custom_iou.item())
        
        del inputs
        del targets
        del predictions
    
    
    iou_tensor = torch.FloatTensor(iou_accuracies)
    pixel_tensor = torch.FloatTensor(pixel_accuracies)
    custom_iou_tensor = torch.FloatTensor(custom_iou_accuracies)
    
    print("Test Dataset Accuracy")
    print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}, Custom IoU Accuracy: {custom_iou_tensor.mean():.4f}")
    return iou_tensor.mean(), pixel_tensor.mean(), custom_iou_tensor.mean()

 
