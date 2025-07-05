import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import argparse
import yaml
from tqdm import tqdm
from datetime import timedelta
from dataset.load_data import TimeSeriesDataset
from models.model import CNNModel, LSTMModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle
import numpy as np
from utils import pre_process, calculate_class_weights
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='path to config file')
    return parser.parse_args()

def save_checkpoint(epoch, model, optimizer, best_loss, is_best, save_dir):
    # Saving weights after the end of each epoch.
    last_checkpoint_path = os.path.join(save_dir, 'last.pth')
    best_checkpoint_path = os.path.join(save_dir, 'best.pth')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }, last_checkpoint_path)
    print(f'Last checkpoint saved to {last_checkpoint_path}')

    if is_best:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, best_checkpoint_path)
        print(f'Best checkpoint saved to {best_checkpoint_path}')


def train(model, optimizer, batch_size, train_loader, val_loader,
          device, num_epochs, criterion, save_dir, start_epoch):

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    train_steps = len(train_loader.dataset) // batch_size
    val_steps = len(val_loader.dataset) // batch_size
    H = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    print('\nstart training the model...')

    start_time = time.time()
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    early_stopping_patience = 10
    patience = early_stopping_patience


    for epoch in range(start_epoch, num_epochs):
        ### Training Phase ###
        model.train()
        total_train_loss = 0

        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader)

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs, _ = model(inputs)

            loss = criterion(outputs, labels)
            # weighted loss function: https://chatgpt.com/c/6861800b-fcd8-8004-8d93-1d9ad01bbd29

            # loss = custom_weighted_bce(outputs, labels)


            # Calculate accuracy
            #accuracy = accuracy_score(y_true, y_pred)
            # preds = (outputs >= 0.5).float()  # Apply 0.5 threshold for binary classification
            # correct_train += (preds == labels).sum().item()
            # total_train += labels.size(0)


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            total_train_loss += loss.item()
            pbar.set_description(f'Epoch {epoch + 1} / {num_epochs} Loss: {loss.item():.4f}')

        avg_train_loss = total_train_loss / train_steps
        # train_accuracy = correct_train / total_train
        H['train_loss'].append(avg_train_loss)
        # H['train_acc'].append(train_accuracy)

        ### Validation Phase ###
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs, _ = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                # val_loss = multiclass_precision_focused_loss(val_outputs, val_labels)
                # val_loss = custom_weighted_bce(val_outputs, val_labels)

                # Calculate accuracy
                # val_preds = (val_outputs >= 0.5).float()
                # correct_val += (val_preds == val_labels).sum().item()
                # total_val += val_labels.size(0)

                total_val_loss += val_loss.item()


        avg_val_loss = total_val_loss / val_steps
        # val_accuracy = correct_val / total_val
        H['val_loss'].append(avg_val_loss)
        # H['val_acc'].append(val_accuracy)

        # Update best metrics and save model if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_best = True
            patience = early_stopping_patience
        else:
            is_best = False
            patience -= 1
        print(f'\n==> [Finished epoch {epoch + 1} / {num_epochs}]: Average training loss: {avg_train_loss:05f}. '
              f'Average validation loss: {avg_val_loss:05f}  '
              #f'Training accuracy: {train_accuracy:.4f}, '
              #f'Validation accuracy: {val_accuracy:.4f}'
              )



        save_checkpoint(epoch + 1, model, optimizer, best_val_loss, is_best, save_dir)
        scheduler.step(round(avg_val_loss, 4))
        print(f'Current learning rate: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

        if patience == 0:
            print(f'Early stopping at epoch  {epoch + 1}')
            break

    elapsed = time.time() - start_time
    print(f'elapsed time:  {timedelta(seconds=elapsed)}')

    # Plot and save losses
    plt.figure()
    plt.plot(H['train_loss'], label='train loss')
    plt.plot(H['val_loss'], label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Loss over Epochs")
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()

    # # Plot and save accuracies
    # plt.figure()
    # plt.plot(H['train_acc'], label='Train Accuracy')
    # plt.plot(H['val_acc'], label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend(loc='lower right')
    # plt.title("Accuracy over Epochs")
    # plt.savefig(os.path.join(save_dir, 'accuracies.png'))
    # plt.close()


def load_best_weights(model, checkpoint_dir):
    # Load best weights.
    weights_dir = os.path.join(checkpoint_dir, 'best.pth')
    model.load_state_dict(torch.load(weights_dir)['model_state_dict'])
    torch.save(model, os.path.join(checkpoint_dir, 'best_model.pt'))
    return model

def evaluation(val_generator, device, model, checkpoint_dir, criterion, set_type='evaluation'):
    '''
    evaluate the model on the validation dataset.
    Args:
        val_generator: (dataset generator) validation dataset generator.
        device: (torch.device).
        model: (model).
        checkpoint_dir: (str) weights directory.
        criterion: (torch.nn.modules.loss) loss function.

    Returns:
    avg_val_loss: (float) average validation loss.
    '''

    # Validation dataloader.
    val_loader = DataLoader(dataset=val_generator, batch_size=1, shuffle=False)

    # initialize the first element of key list.
    key_list = []
    key_list.append('_gt')
    key_list.append('_pred')

    eval_dict = {k: [] for k in key_list}

    total_val_loss = 0
    val_steps = len(val_loader.dataset)
    correct_predictions = 0

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc='evaluation'):
            x = x.to(device)
            y = y.to(device)
            pred, features = model(x)

            val_loss = criterion(pred, y)
            total_val_loss += val_loss.item()

            eval_dict['_gt'].append(y.cpu().numpy()[0].argmax())
            eval_dict['_pred'].append(pred.cpu().numpy()[0].argmax())


    df = pd.DataFrame.from_dict(eval_dict)
    output_dir = os.path.join(checkpoint_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    if set_type == 'test':
        df.to_csv(os.path.join(output_dir,'test.csv'), index=False)
    elif set_type == 'evaluation':
        df.to_csv(os.path.join(output_dir, 'evaluation.csv'), index=False)
    elif set_type == 'train':
        df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    avg_val_loss = total_val_loss / val_steps
    accuracy = correct_predictions / val_steps
    return avg_val_loss, accuracy, os.path.join(output_dir,'evaluation.csv')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)

    # Create checkpoint directory.
    save_dir = os.path.join('../experiments/', config_data['DIRECTORIES']['OUTPUT_FOLDER'])
    os.makedirs(save_dir, exist_ok=True)


    # Hyperparameters.
    learning_rate = float(config_data['HYPER_PARAMS']['LEARNING_RATE'])
    num_epochs = config_data['HYPER_PARAMS']['NUM_EPOCHS']
    batch_size = config_data['HYPER_PARAMS']['BATCH_SIZE']

    # Read the dataset.
    dataset_dir = config_data['DIRECTORIES']['DATASET_DIR']

    X, y = pre_process(dataset_dir)
    # np.save('X.npy', X)
    # np.save('y.npy', y)
    # preprocessing takes too much time, so I saved X and y locally, and reload them.
    # X = np.load('X.npy')
    # y = np.load('y.npy')


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    class_weights = calculate_class_weights(y_train)
    print("Class Weights:", class_weights) #[0.00127307 0.09508928 0.00979841 0.89383923]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_generator = TimeSeriesDataset(X_train, y_train)
    val_generator = TimeSeriesDataset(X_val, y_val)

    num_input = 12

    # Save updated config to the experiment folder
    with open(os.path.join(save_dir, "config.yaml"), "w") as file:
        yaml.dump(config_data, file)


    train_loader = DataLoader(dataset=train_generator, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_generator, batch_size=batch_size, shuffle=False)


    start_epoch = 0
    # Model
    # model = CNNModel(num_input, sequence_length, output_size).to(device)
    model = LSTMModel(num_input).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    train(model, optimizer, batch_size, train_loader, val_loader,
          device, num_epochs, criterion, save_dir, start_epoch)

    model = load_best_weights(model, save_dir)


    # Evaluate the model.
    val_loss, accuracy, evaluation_file_path = evaluation(val_generator, device, model, save_dir, criterion)


    print(f'accuracy {accuracy}')
    print(f'validation loss {val_loss}')

if __name__ == '__main__':
    main()