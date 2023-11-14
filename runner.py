import json
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile, UnidentifiedImageError
import PIL
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import warnings
import os
import torch

warnings.filterwarnings("ignore")

def AccessDatabase():
    with open('database_dump.txt', 'r') as f:
        contents = f.read()

    data = json.loads(contents)

    sorted_data = sorted(data.items())

    new_list = [[value[0][2:], value[1]] for key, value in sorted_data]

    categories = ["Armory", "Crypto", "Drugs", "Electronics", "Financial", "Gambling", "Hacking", "Pornography", "Violence", "Legal"]
    new_data_structure = []

    for item in new_list:
        link = item[0].replace('/', '_').replace(':', '_').strip()
        scores = item[1]
        scores_list = [int(scores[i*3:i*3+3])/100 for i in range(len(categories))]
        new_data_structure.append((link, scores_list))

    return new_data_structure

def CreatePicFolder(new_data_structure):
    # Define the path to the 'pic' subfolder
    pic_folder_path = r'C:\Users\Ragav\Desktop\Prophecy\COLLEGE\CLUBs\SPIDER\NOCAINE\Dataset\pic'

    # Get a list of all files in the 'pic' subfolder
    pic_files = os.listdir(pic_folder_path)

    pics = set()

    # Iterate over the list 'new_data_structure'
    for data in new_data_structure:
        # Get the first element of the tuple
        prefix = data[0]
        # Check each file in the 'pic' subfolder
        for filename in pic_files:
            # If the filename starts with the prefix, print it
            if filename.startswith(prefix):
                    pics.add((filename, tuple(data[1])))

    pics = list(pics)

    return pic_folder_path, pics

class ImageDataset(Dataset):
    def __init__(self, pic_folder_path, pics, transform):
        self.pic_folder_path = pic_folder_path
        self.pics = pics
        self.transform = transform

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        filename, _ = self.pics[idx]
        img_path = os.path.join(self.pic_folder_path, filename)
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except (UnidentifiedImageError, OSError):
            print(f"Cannot identify image file {img_path}. Skipping.")
            # Return a tensor of zeros with the same shape as your images
            image = torch.zeros(3, 224, 224)  # Adjust the shape as needed
        return image

class TupleDataset(Dataset):
    def __init__(self, pics):
        self.pics = pics

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        _, data = self.pics[idx]
        # Apply softmax to the data tensor
        data = torch.nn.functional.softmax(torch.tensor(data), dim=0)
        return data


new_data_structure = AccessDatabase()

pic_folder_path, pics = CreatePicFolder(new_data_structure)

# Define the transform to be applied to each image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


image_dataset = ImageDataset(pic_folder_path, pics, transform)
tuple_dataset = TupleDataset(pics)

image_dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True)
tuple_dataloader = DataLoader(tuple_dataset, batch_size=32, shuffle=True)

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the saved model
model_path = "resnet50_model.pt"

# Check if a saved model exists
if os.path.exists(model_path):
    # Load the saved model
    model = torch.load(model_path)
    print("Saved model loaded successfully.")
else:
    # Define the ResNet50 model
    print("No saved model found. Loading ResNet50...")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)


# Define the loss function and optimizer
criterion = nn.MSELoss()
# criterion = criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Create an iterator for the tuple_dataloader
tuple_dataloader_iter = iter(tuple_dataloader)

# Train the model
for epoch in range(10):
    for i, (images, labels) in enumerate(zip(image_dataloader, tuple_dataloader_iter)):  

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print statistics
        print(f"Epoch: {i+1}, Loss: {loss.item():.4f}")


print('Finished Training')

# Save the model
torch.save(model, model_path)






