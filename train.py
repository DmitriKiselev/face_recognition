#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import torch
import torchvision
import time

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir_to_train',
                        help='Directory of train set.', type=str)
    parser.add_argument('--dir_to_val', type=str,
                        help='Directory of validation set.')
    parser.add_argument('--num_epochs',  type=int,
                        help='Number of epochs.')
    parser.add_argument('--resize_size', type=int, nargs=2,
                        default=[224, 224],
                        help='Set size for resizing image')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--path_to_save_1', type=str,
                        help='Path for saving model with best')
    parser.add_argument('--dir_writer', type=str,
                        help='Directory for saving graphs ')
    return parser.parse_args()


def train_model(model,
                criterion,
                optimizer,
                train_loader,
                val_loader,
                num_epochs,
                path_save_model_1,
                dir_writer,
                is_inception=False):

    """

    Parameters
    ----------
    model : torch model
        ResNet18 model for train.
    criterion : function loss
        Cross Entropy Loss
    optimizer: torch opimizer
        Torch optimizer
    train_loader: DataLoader
        Train dataloader
    val_loader: DataLoader
        Validation Dataloader
    num_epochs: int
        Number epochs for train
    path_save_model_1:
        Path for saving model with best train accuracy
    dir_writer:
        Direction for saving graphs
    is_inception

    Returns
    -------

    """
    writer = SummaryWriter(dir_writer)
    since = time.time()
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_train_prec_score = 0.0
    best_val_prec_score = 0.0
    best_train_recall_score = 0.0
    best_val_recall_score = 0.0
    min_train_loss = np.inf
    min_val_loss = np.inf

    data_loaders = {'train': train_loader,
                    'val': val_loader}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        cur_metrics = {'train': [], 'val': []}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            pr_sc = []
            rl_sc = []

            for inputs, labels in tqdm(data_loaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds = preds.to('cpu')
                labels = labels.to('cpu')
                pr_sc.append(precision_score(labels, preds, average='binary'))
                rl_sc.append(recall_score(labels, preds, average='binary'))

            cur_metrics[phase].append(
                running_loss / len(data_loaders[phase].dataset))
            cur_metrics[phase].append(
                running_corrects.double() / len(data_loaders[phase].dataset))
            cur_metrics[phase].append(sum(pr_sc) / len(pr_sc))
            cur_metrics[phase].append(sum(rl_sc) / len(rl_sc))
            print('{} Loss: {:.4f} Acc: {:.4f} Presicion score: '
                  '{:.4f} Recall score: {:.4f}'.format(
                                                       phase,
                                                       cur_metrics[phase][0],
                                                       cur_metrics[phase][1],
                                                       cur_metrics[phase][2],
                                                       cur_metrics[phase][3]
                                                     ))

        save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }

        # saving models with best results
        if cur_metrics['train'][1] > best_train_acc:
            best_train_acc = cur_metrics['train'][1]
            torch.save(save, path_save_model_1 + 'model_best_train_accuracy'
                                                 '.pth')
        if cur_metrics['val'][1] > best_val_acc:
            best_val_acc = cur_metrics['val'][1]
            torch.save(save, path_save_model_1 + 'model_best_val_accuracy.pth')
        if cur_metrics['train'][0] < min_train_loss:
            min_train_loss = cur_metrics['train'][0]
            torch.save(save, path_save_model_1 + 'model_best_train_loss.pth')
        if cur_metrics['val'][0] < min_val_loss:
            min_val_loss = cur_metrics['val'][0]
            torch.save(save, path_save_model_1 + 'model_best_val_loss.pth')
        if cur_metrics['train'][2] > best_train_prec_score:
            best_train_prec_score = cur_metrics['train'][2]
            torch.save(save, path_save_model_1 + 'model_best_train_prec_score.'
                                                 'pth')
        if cur_metrics['val'][2] > best_val_prec_score:
            best_val_prec_score = cur_metrics['val'][2]
            torch.save(save, path_save_model_1 + 'model_best_val_prec_score.'
                                                 'pth'
                       )
        if cur_metrics['train'][3] > best_train_recall_score:
            best_train_recall_score = cur_metrics['train'][3]
            torch.save(save, path_save_model_1 + 'model_best_train_recall_'
                                                 'score.pth')
        if cur_metrics['val'][3] > best_val_recall_score:
            best_val_recall_score = cur_metrics['val'][3]
            torch.save(save, path_save_model_1 + 'model_best_val_recall_score'
                                                 '.pth')

        writer.add_scalars('Loss', {'Train': cur_metrics['train'][0],
                                    'Valid': cur_metrics['val'][0]}, epoch)
        writer.add_scalars('Accuracy', {'Train': cur_metrics['train'][1],
                                        'Valid': cur_metrics['val'][1]}, epoch)
        writer.add_scalars('Precision score', {'Train': cur_metrics['train']
                                               [2],
                                               'Valid': cur_metrics['val']
                                               [2]}, epoch)
        writer.add_scalars('Recall score', {'Train': cur_metrics['train']
                                            [3],
                                            'Valid': cur_metrics['val']
                                            [3]}, epoch)
        writer.flush()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_val_acc))
    writer.close()
    return model


def main():
    """Application entry point."""
    args = get_args()

    train_transforms = transforms.Compose([transforms.Resize((
                                           args.resize_size[0],
                                           args.resize_size[1])),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(10),
                                           transforms.ToTensor(),
                                           transforms.Normalize
                                           ((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])

    test_transforms = transforms.Compose([transforms.Resize((
                                          args.resize_size[0],
                                          args.resize_size[1])),
                                          transforms.ToTensor(),
                                          transforms.Normalize
                                          ((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])

    data_paths = {'train': args.dir_to_train, 'val': args.dir_to_val}

    train_dataset = ImageFolder(data_paths['train'], train_transforms)
    val_dataset = ImageFolder(data_paths['val'], test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # model definition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(pretrained=False, num_classes=2)
    weights = torchvision.models.resnet18(pretrained=True).state_dict()
    del weights['fc.weight']
    del weights['fc.bias']
    model.load_state_dict(weights, strict=False)
    model.to(device)
    #weights = [0.7, 0.3]
    #class_weights = torch.FloatTensor(weights).cuda()
    loss_fn = nn.CrossEntropyLoss()#weight=class_weights
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    train_1 = train_model(model=model, criterion=loss_fn,
                          optimizer=optimizer,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          num_epochs=args.num_epochs,
                          path_save_model_1=args.path_to_save_1,
                          dir_writer=args.dir_writer)

    return


if __name__ == '__main__':
    main()
