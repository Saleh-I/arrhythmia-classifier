import torch
from torch.utils.data import Dataset, DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        '''

        Args:
            X: (numpy array) (140654, 256, 12) ; 140654: number of samples. 256: window size. 12: leads.
            y: (numpy array)  (140654,) labels
        '''

        self.X = X
        self.y = y

    def __len__(self):
            # Total number of samples we can generate
            return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        label = self.y[idx] # Classes: ['+' 'A' 'B' 'F' 'N' 'Q' 'R' 'S' 'V' 'j' 'n']


        if label in ['N', 'R', 'L', 'n', 'B']:  # Normal
            output_label = [1.0, 0.0, 0.0, 0.0] #'N'
        elif label in ['A', 'S', 'j']:  # Supraventricular
            output_label = [0.0, 1.0, 0.0, 0.0] #'SVEB'
        elif label in ['V']:  # Ventricular
            output_label = [0.0, 0.0, 1.0, 0.0] #'VEB'
        else:
            output_label = [0.0, 0.0, 0.0, 1.0] #'Other'

        return x, torch.tensor(output_label, dtype=torch.float32)




