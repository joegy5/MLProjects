import torch, pandas, ast # type: ignore
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights # type: ignore
from torchvision import transforms
from pathlib import Path

class ImageDataset(Dataset): 
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform: 
            image = self.transform(image) # no need to convert to tensor, transforms.ToTensor() handles that
        return image, torch.tensor(self.labels[index], dtype=torch.float32)


def compute_dataset_statistics(image_paths, labels):
    preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor() # turns input into tensor and also divides each pixel value by 255
        ])

    image_dataset = ImageDataset(image_paths, labels, transform=preprocess)
    print(len(image_dataset))
    image_dataloader = DataLoader(image_dataset, batch_size=32, num_workers=NUM_WORKERS)
    mean, std = torch.zeros(3), torch.zeros(3)
    for image_batch, _ in image_dataloader:
        batch_size = image_batch.size(0)
        # image_batch current shape is (batch_size, num_channels, height, width) due to transforms.toTensor()
        # need to reshape it to (batch_size, num_channels, num_pixels) (num_pixels = height * width)
        curr_batch = image_batch.reshape(batch_size, image_batch.size(1), -1)
        # .mean(2) / .std(2) takes mean/std for each image in batch --> last dimension gets collapsed
        # .sum(0) sums means of all images across batch --> collapses first dimension
        # results are just 1D tensors of length 3 representing mean/std across all images in the batch for each of the 3 channels (red, green, blue)
        mean += curr_batch.mean(2).sum(0)
        std += curr_batch.std(2).sum(0)
    
    # average means and stds across all batches so that CNN works with consisten mean/std 
    # for ALL batches. Using different mean and std for each batch introduces inconsistency in the model, negatively affecting its learning
    mean /= len(image_paths)
    std /= len(image_paths)
    return mean, std


def create_train_val_dataloaders(image_paths, labels, transform=None):
    train_size = int(0.8*len(image_paths))
    val_size = len(image_paths) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=ImageDataset(image_paths, labels, transform),
        lengths=[train_size, val_size]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    return train_dataloader, val_dataloader


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4
NUM_EPOCHS = 10
NUM_CLASSES = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

dataset_file_path = Path("FundusDataset/full_df.csv")
df = pandas.read_csv(dataset_file_path, usecols=["labels", "filename"])
df["labels"] = df["labels"].apply(ast.literal_eval)
image_directory = Path("FundusDataset/ODIR-5K/ODIR-5K/Training Images")

labels = df["labels"].tolist()
image_paths = [image_directory / filename for filename in df["filename"]]

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Unfreeze 4th layer block
for param in model.layer4.parameters():
    param.requires_grad = True
# Replace the last fully-connected layer
num_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features=num_features, out_features=NUM_CLASSES)
for param in model.fc.parameters():
    param.requires_grad = True
model = model.to(DEVICE)

update_params = list(model.layer4.parameters()) + list(model.fc.parameters())
optimizer = torch.optim.Adam(update_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# dealing with multi-label classification --> use BCEWithLogitsLoss, handles each binary loss independently
# and removes need for applying softmax before passing into loss function (takes in logits directly)
bce_logits_loss = torch.nn.BCEWithLogitsLoss()

mean, std = compute_dataset_statistics(image_paths, labels)
image_preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])
train_dataloader, val_dataloader = create_train_val_dataloaders(image_paths, labels, transform=image_preprocessing)

# Apply transfer learning to the model
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for train_batch_images, train_batch_labels in train_dataloader:
        print(train_batch_images.size(0))
        optimizer.zero_grad()
        image_batch, label_batch = train_batch_images.to(DEVICE), train_batch_labels.to(DEVICE) 
        
        model_outputs = model(image_batch)
        loss = bce_logits_loss(model_outputs, label_batch)
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * image_batch.size(0)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(image_paths)}")

weights_file_path = Path("C:/Users/joelp/Downloads/OptiClarityFundusModels/resnet50_weights.pth")
torch.save(model.state_dict(), weights_file_path)
# model2.load_state_dict(torch.load(weights_file_path)) to load in weights to model2





