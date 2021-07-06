import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import torch.optim as optim;

from torch.utils.data import DataLoader;
import torchvision;

import CIFAR10_utils as CIFAR;

from torch.autograd import Variable;

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(6, 6, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(6, 6, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(6, 6, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(6, 6, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=2, padding=1),nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return x, y;
    
    
    
    
    

def createNetwork():
    autoencoder = Autoencoder()
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda();
    return autoencoder;

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def trainNetwork(network, data_loader, num_epochs = 20, learning_rate = 0.001):
    with torch.cuda.device(0):
        criterion = nn.BCELoss();
        optimizer = optim.Adam(network.parameters());

        for epoch in range(10):
            running_loss = 0.0;
            for i, (inputs, _) in enumerate(data_loader, 0):
                inputs = get_torch_vars(inputs);

                encoded, outputs = network(inputs);
                loss = criterion(outputs, inputs);
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                running_loss += loss.data
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000));
                    running_loss = 0.0;
        

def project(im, network, num_epochs = 200, learning_rate = 0.1):
    im = torch.from_numpy(CIFAR.VectorToImage(im)).float();
    im = im.cuda();
    inp = im.clone();
    inp = inp.cuda();
    inp.requires_grad_(True);
    
    optimizer = torch.optim.Adam([inp], lr=learning_rate);

    for i in range(1, num_epochs + 1):
        optimizer.zero_grad();
        out = network(inp.unsqueeze(0))[1];
        loss = torch.norm(im.add(-out));
        loss.backward();
        optimizer.step();
        
    return CIFAR.ImageToVector(network(inp.unsqueeze(0))[1].detach().cpu().numpy());

def save_network(network, filename):
    torch.save(network.state_dict(), filename);
    
def load_network(network, filename):
    return network.load_state_dict(torch.load(filename));