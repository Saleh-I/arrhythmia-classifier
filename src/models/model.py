import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu_conv1 = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu_conv2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)


        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu_conv3 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Adjust the size for the first fully connected layer based on the sequence length
        reduced_sequence_length = sequence_length // 2 // 2  # Two pooling layers halve the size twice
        self.fc1 = nn.Linear(128 * reduced_sequence_length, 256)
        self.relu_fc1 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, sequence_length)
        x = self.relu_conv1(self.conv1(x))
        x = self.pool1(self.relu_conv2(self.conv2(x)))
        x = self.pool2(self.relu_conv3(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        features = self.fc2(x)
        # x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc2(features)
        x = self.fc3(x)
        return x, features



import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=4, num_layers=1):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)


        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()


        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Get the last time step output
        x = self.relu1(self.fc1(last_output))
        x = self.relu2(self.fc2(x))

        out = self.fc3(x)
        # out = self.sigmoid(self.fc3(x))

        return out, None

