import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import os

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定訓練參數
batch_size = 8
learning_rate = 0.001
epochs = 100
threshold = 0.95  # 設定閥值，可根據需要調整

# 定義簽名辨識CNN模型
class SignatureCNN(nn.Module):
    def __init__(self, num_classes):
        super(SignatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes+1)  # 多分類問題，輸出等於類別數

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 載入數據集並進行預處理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# 檢查是否已經存在已儲存的模型
if os.path.exists('signature_model.pth'):
    # 如果模型文件存在，則載入模型參數
    train_dataset = ImageFolder(root='train', transform=transform)
    num_classes = len(train_dataset.classes)  # 獲取類別數
    model = SignatureCNN(num_classes)  # 創建一個新的模型實例
    model.load_state_dict(torch.load('signature_model.pth'))  # 載入模型參數
    model.eval()
    print("已載入已儲存的模型")
else:
    train_dataset = ImageFolder(root='train', transform=transform)
    num_classes = len(train_dataset.classes)  # 獲取類別數

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 創建模型、損失函數和優化器
    model = SignatureCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練模型
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # 儲存已訓練的模型
    torch.save(model.state_dict(), 'signature_model.pth')
    print("已儲存已訓練的模型")

# 使用訓練好的模型進行預測
test_image = Image.open('test/test_image.png').convert('L')
test_image = transform(test_image).unsqueeze(0)
with torch.no_grad():
    prediction = model(test_image)
    probabilities = torch.softmax(prediction, dim=1)
    max_probability, predicted_label = torch.max(probabilities, 1)
    predicted_class = train_dataset.classes[predicted_label.item()]
    
    if max_probability.item() >= threshold:
        print(f'This signature belongs to the class: {predicted_class} with a probability of {max_probability.item():.4f}')
    else:
        print("This signature does not belong to any known class.")

# debug
test_image = Image.open('test/test_image.png').convert('L')
test_image = transform(test_image).unsqueeze(0)
test_image = test_image.to(device)  
with torch.no_grad():
    model = model.to(device)
    prediction = model(test_image)
    probabilities = torch.softmax(prediction, dim=1)

    _, predicted_labels = torch.topk(probabilities, k=num_classes)
    
    for i in range(num_classes):
        predicted_class = train_dataset.classes[predicted_labels[0][i].item()]
        probability = probabilities[0][predicted_labels[0][i]].item()
        print(f'Class: {predicted_class}, Probability: {probability:.4f}')