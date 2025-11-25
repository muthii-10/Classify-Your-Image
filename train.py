#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
from torchvision import models
import os
from input_args import get_input_args
from load_and_prep import load_prep_data

def main():
    args = get_input_args()
    train_loader, valid_loader, test_loader = load_prep_data(args.data_dir)

    # Choose architecture and create model
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print(f"Architecture {args.arch} not recognized, using vgg16.")
        model = models.vgg16(pretrained=True)

    # Define a new classifier to fit  number of classes
    classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(args.hidden_units, len(load_and_prep()[2].classes)),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier

    # Move model to GPU if available and required
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)

    # Define optimizer and criterion
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Training loop
    epochs = args.epochs
    steps = 1
    print_every = 10

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero out the gradients
            optimizer.zero_grad()

            logps = model.forward(inputs) # Forward pass
            loss = criterion(logps, labels) # Calculate the log probabilities
            loss.backward() # Do gradient descent
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0.0

                # Do validation on the validation set
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate Validation Accuracy
                        probs = torch.exp(logps)
                        top_p, top_class = probs.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs} .." 
                      f"Training loss: {(running_loss/print_every):.3f}.."
                      f"Validation loss: {valid_loss/len(valid_loader):.3f}.." 
                      f"Validation accuracy: {accuracy/len(valid_loader):.3f} ")
                
                running_loss = 0
                model.train()

    # Save checkpoint
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    checkpoint = {
        'model': model,
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': load_and_prep()[2].class_to_idx
    }
    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    main()




