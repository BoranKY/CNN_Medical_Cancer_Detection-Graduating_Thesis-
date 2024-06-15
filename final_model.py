################################################
# Import Libraries
################################################

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score
################################################
# RESNET-34
################################################
class SimpleResNet34(nn.Module):
    def __init__(self,num_classes=2):
        super(SimpleResNet34,self).__init__()
        # Pretrained ResNet34 modelini yükleyelim
        self.resnet34 = models.resnet34(pretrained=True)

        # Modelin son fully connected katmanını değiştiriyoruz
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs,num_classes)

    def forward(self,x):
        return self.resnet34(x)

################################################
# Custom Dataloader
################################################
class CustomMRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['negatif', 'pozitif']
        self.samples = self.get_samples()

    def get_samples(self):
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for patient_folder in os.listdir(class_path):
                patient_path = os.path.join(class_path, patient_folder)
                for image_name in os.listdir(patient_path):
                    image_path = os.path.join(patient_path, image_name)
                    samples.append((image_path, self.classes.index(class_name)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

################################################
# Transforms Process
################################################

# New transforms mean and std
train_mean = [0.0691, 0.0691, 0.0691]
train_std = [0.1308, 0.1308, 0.1308]

test_mean = [0.0602, 0.0602, 0.0602]
test_std = [0.1219, 0.1219, 0.1219]



transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean, std=train_std)
])

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=test_mean, std=test_std)
])

################################################
# Load datasets
################################################
train_dir = "train_image_directory"
train_set = CustomMRDataset(train_dir, transform=transforms_train)

test_dir = "test_image_directory"
test_set = CustomMRDataset(test_dir, transform=transforms_test)

################################################
# Default Dataloader for test ve train loader
################################################

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)

################################################
# Initialize the model
################################################
model = SimpleResNet34()

################################################
# Define loss function and optimizer
################################################

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, verbose=True)

# Choosing the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

################################################
# Training Loop
################################################
num_epochs = 15
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
roc_aucs = []
all_y_pred = []
all_y_true = []
all_outputs = []
specificities = []
ppvs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            all_outputs.extend(outputs.cpu().numpy())
            _, predicted = torch.max(outputs, 1)
            all_y_pred.extend(predicted.cpu().numpy())
            all_y_true.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    all_outputs_np = np.array(all_outputs)
    positive_class_probabilities = torch.softmax(torch.tensor(all_outputs_np), dim=1)[:, 1].numpy()
    roc_auc = roc_auc_score(all_y_true, positive_class_probabilities)
    roc_aucs.append(roc_auc)

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Calculate specificity and positive predictive value (PPV)
    specificity = TN / (TN + FN)
    ppv = TP / (TP + FP)
    specificities.append(specificity)
    ppvs.append(ppv)

    sensitivity = recall_score(all_y_true, all_y_pred, average='macro')

    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, PPV: {ppv:.4f}")

    scheduler.step(test_loss)



# Post-training calculations
cm = confusion_matrix(all_y_true,all_y_pred)

print("Confusion Matrix:")
print(cm)

# Heatmap for normalized confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap

# Save the figure
plt.savefig("Confusion_matrix.png")
plt.close()


# Normalized confusion matrix calculation
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("Normalized Confusion Matrix:")
print(cm_normalized)


# Heatmap for normalized confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap

# Save the figure
plt.savefig("normalized_confusion_matrix.png")
plt.close()


################################################
# Printing the data table
################################################

tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp


accuracy = (tp + tn) / total
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
specificity = tn / (tn + fn)
ppv = tp / (tp + fp)

# Organize and display results in a DataFrame
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "PPV"],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        specificity,
        ppv
    ]
})

print(results)


################################################
# Plotting as a bar graphic
################################################

fig, ax = plt.subplots(figsize=(12, 8))
results.plot(kind='barh', x='Metric', y='Value', ax=ax, color='skyblue', legend=None)
ax.set_xlabel('Value')
ax.set_title('Training Results')

# Increase visibility of values on x-axis
for i in ax.patches:
    ax.text(i.get_width() + 0.02, i.get_y() + 0.5, str(round((i.get_width()), 2)), fontsize=10, color='dimgrey')

# Save the Bar Graph
plt.savefig('Graph.png')
plt.close()

################################################
# Plotting as a table
################################################

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=results.values, colLabels=results.columns, cellLoc = 'center', loc='center')
# Save the Table
plt.savefig('Table.png')
plt.close()


################################################
# Plotting Loss & Accuracy Values
################################################

# Plotting Loss
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Accuracy
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot ROC-AUC
fpr, tpr, _ = roc_curve(all_y_true, positive_class_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Plotting PPV
plt.plot(ppvs, label='PPV')
plt.xlabel('Epoch')
plt.ylabel('Positive Predictive Value')
plt.legend()
plt.show()

# Plotting Specificity
plt.plot(specificities, label='Specificity')
plt.xlabel('Epoch')
plt.ylabel('Specificity')
plt.legend()
plt.show()










