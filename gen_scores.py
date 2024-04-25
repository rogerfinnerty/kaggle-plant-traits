import torch
from torch import nn
from torchvision import models
from torchvision import transforms

from tqdm import tqdm

import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image

import numpy as np

ipt = '../planttraits2024'
opt = 'out'
netpth = '../resnet50.a1_in1k_best_model2_2.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

class MultiModalNetwork(nn.Module):
    def __init__(self, num_classes=6):
        super(MultiModalNetwork, self).__init__()
        # 画像のためのバックボーンモデル
        self.efficientnet_backbone = models.efficientnet_b0(pretrained=False)
        self.efficientnet_backbone = nn.Sequential(*list(self.efficientnet_backbone.children())[:-1])  # 最後の分類層を除外
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(0.5)

        # 数値データのためのレイヤー
        self.dense1 = nn.Linear(155, 326)
        self.dense2 = nn.Linear(326, 64)
        self.dropout2 = nn.Dropout(0.5)

        # 結合後のレイヤー
        self.concat_dense = nn.Linear(1280 + 64, num_classes)

    def forward(self, image_input, numerical_input):
        # 画像のためのパス
        x1 = self.efficientnet_backbone(image_input)
        x1 = self.pooling(x1).view(-1, x1.shape[1])  # Flatten
        x1 = self.dropout1(x1)

        # 数値データのためのパス
        x2 = nn.functional.relu(self.dense1(numerical_input))
        x2 = nn.functional.relu(self.dense2(x2))
        x2 = self.dropout2(x2)

        # 結合
        x = torch.cat((x1, x2), dim=1)

        # 最終的な分類
        x = self.concat_dense(x)

        return x


# Assuming that the train.csv is already loaded into the environment
# and we have a PyTorch model defined and trained called 'MyModel'.
# Replace 'MyModel' and its parameters with your actual model.

# Define the dataset
class PlantTraitsDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        self.transform = transform
        self.train = train
        # Load the dataset
        df = pd.read_csv(csv_file)
        # Ignore columns 'FQ' (156) to 'FT' (173)
        self.ids = df['id']
        self.features = df.iloc[:, 1:156]  # Select columns from 'A' to 'FP' (0 to 155)
        self.features = (self.features - self.features.mean(axis=0)) / self.features.std(axis=0)

        self.labels = df.iloc[:, 158:164]  # Select columns for the labels 'FI' to 'FN' (158 to 163)
        self.std = self.labels.std(axis=0)
        self.mean = self.labels.mean(axis=0)
        self.labels = (self.labels - self.mean) / self.std
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Load the image
        if self.train:
            image_path = os.path.join(ipt, 'train_images', str(self.ids.iloc[idx]) + '.jpeg')
        else:
            image_path = os.path.join(ipt, 'test_images', str(self.ids.iloc[idx]) + '.jpeg')
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        # Load the features and labels
        features = torch.tensor(self.features.iloc[idx].values.astype('float32'))
        if self.train:
            labels = torch.tensor(self.labels.iloc[idx].values.astype('float32'))
            return image, features, labels
        else:
            return image, features

# transform
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4482, 0.4525, 0.3360], std=[0.1086, 0.0971, 0.1172]),
])
# transform
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4482, 0.4525, 0.3360], std=[0.1086, 0.0971, 0.1172]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 高速化
torch.backends.cudnn.benchmark = True
# 再現性が担保される
torch.backends.cudnn.deterministic = False

# Load the trained model
model = MultiModalNetwork(num_classes=6)
model = model.to(device)
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(netpth))

model.eval()
dataset = PlantTraitsDataset(os.path.join(ipt, 'train.csv'), transform=train_transform, train=True)
std = dataset.std
mean = dataset.mean
testset = PlantTraitsDataset(os.path.join(ipt, 'test.csv'), transform=test_transform, train=False)
test_loader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

# Make predictions
predictions = []
with torch.no_grad():
    with tqdm(total=len(test_loader)) as pbar:
        for i, (image, numerical) in enumerate(test_loader):
            image = image.to(device)
            numerical = numerical.to(device)
            output = model(image, numerical)
            pbar.update(1)
            predictions.append(output.detach().cpu().numpy())

means = np.array(mean)
stds = np.array(std)
print(f'mean:{means} stds:{stds}')
print(stds.shape)
predictions = np.concatenate(predictions, axis=0)
print(predictions.shape)
predictions = (predictions * stds ) + means

# Assuming the 'id' column is the first column in the train.csv
ids = pd.read_csv(os.path.join(ipt, 'sample_submission.csv'))['id']

# Create the submission DataFrame
submission_df = pd.DataFrame(predictions, columns=['X4', 'X11', 'X18', 'X50', 'X26', 'X3112'])
submission_df.insert(0, 'id', ids)

# Save the submission file
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)
submission_df

# test = pd.read_csv('/kaggle/input/planttraits2024/test.csv')
# train =pd.read_csv('/kaggle/input/planttraits2024/train.csv')

# train = train[train['X4_mean'] >= 0]
# y_columns = [col for col in train.columns if col.endswith('_mean')]
# target = train[y_columns]
# # Rearrange columns in the target DataFrame
# target = target[['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']]

# filtered_train = target[target['X4_mean'] >= 0]
# filtered_train = filtered_train[filtered_train['X11_mean'] < 100]
# filtered_train= filtered_train[filtered_train['X18_mean'] < 50]
# filtered_train= filtered_train[filtered_train['X26_mean'] < 5000]
# filtered_train= filtered_train[filtered_train['X50_mean'] < 10]
# filtered_train[filtered_train['X3112_mean'] < 25000]

# mean_values = filtered_train.mean()
# prediction_columns = ['X4', 'X11', 'X18', 'X50', 'X26', 'X3112']

# submission = pd.DataFrame({'id': test['id']})
# submission[prediction_columns] = mean_values
# submission

# submission_df = submission[prediction_columns] + submission_df[prediction_columns]
# submission_df.insert(0, 'id', ids)

# submission_df

# submission_df.to_csv(submission_path, index=False)