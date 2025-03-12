import torch
import torch.nn as nn

# Classifier
class Classifier(nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(input_features, hidden_features, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_features)  # 添加 Batch Normalization
        self.relu1 = nn.ReLU()

        # add the second hidden layer
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_features // 2)
        self.relu2 = nn.ReLU()

        # add the third hidden layer
        self.fc3 = nn.Linear(hidden_features // 2, num_classes, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)  # apply Batch Normalization
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)  # apply Batch Normalization
        out = self.relu2(out)

        out = self.fc3(out)
        _, class_indices = torch.max(out, dim=1)

        return class_indices
