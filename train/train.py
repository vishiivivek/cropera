import torch
import torchvision
from torchvision import transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F

# Dataset loading
def load_split_train_test(datadir='dataset', test_split=0.1):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(root=datadir, transform=transform)

    dtotal = len(dataset)
    dtest = int(test_split * dtotal)
    dtrain = int(2*test_split * dtotal)

    test_data = torch.utils.data.Subset(dataset, range(dtest))
    train_data = torch.utils.data.Subset(dataset, range(dtest, dtrain))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, testloader, dataset.classes

trainloader, testloader, classes = load_split_train_test()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model, optimizer, criterion

model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, len(classes)),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

model.to(device)

# Train model
epochs = 1
steps = 0
running_loss = 0
print_every = 1
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs} -- "
                  f"Train loss: {running_loss/print_every:.3f} -- "
                  f"Test loss: {test_loss/len(testloader):.3f} -- "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()

device = torch.device("cpu")
model.to(device)

torch.save(model, 'app/classifier.pth')