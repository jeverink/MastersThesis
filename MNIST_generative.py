import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import torch.optim as optim;

from torch.utils.data import DataLoader;
import torchvision;

import MNIST_utils as MNIST;

class Encoder(nn.Module):
    def __init__(self, layercount, bottleneck):
        super(Encoder, self).__init__();
        self.layers = [nn.Linear(in_features = int(28*28/(i+1)), out_features = int(28*28/(i+2))) for i in range(layercount-1)];
        self.layers.append(nn.Linear(in_features = int(28*28/layercount), out_features = bottleneck));
        self.layers = nn.ModuleList(self.layers);
        self.flatten = nn.Flatten();
        
    def forward(self, x):
        x = self.flatten(x);
        for layer in self.layers:
            x = F.relu(layer(x));
        return x;
    
class Decoder(nn.Module):
    def __init__(self, layercount, bottleneck):
        super(Decoder, self).__init__();
        self.layers = [nn.Linear(in_features = int(28*28/(i+2)), out_features = int(28*28/(i+1))) for i in range(layercount-1)];
        self.layers.append(nn.Linear(in_features = bottleneck, out_features = int(28*28/layercount)));
        self.layers = nn.ModuleList(self.layers);
        self.unflatten = nn.Unflatten(1,(1,28,28));
        
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[len(self.layers)-1-i](x));
        x = F.sigmoid(self.layers[0](x));
        x = self.unflatten(x);
        return x;
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__();
        self.encoder = encoder;
        self.decoder = decoder;
        
        
    def forward(self, x):
        x = self.encoder(x);
        x = self.decoder(x);
        return x;
    
    
    
    

def createNetwork(layercount, bottleneck):
    encoder = Encoder(layercount, bottleneck);
    decoder = Decoder(layercount, bottleneck);
    autoEncoder = AutoEncoder(encoder, decoder);
    return (encoder, decoder), autoEncoder;


def trainNetwork(network, data_loader, num_epochs = 20, learning_rate = 0.001):
    criterion = nn.MSELoss();
    optimizer = torch.optim.Adam(network.parameters(), learning_rate);
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0;
        for data in data_loader:
            images, _ = data;
            optimizer.zero_grad();
            outputs = network(images);
            loss = criterion(outputs, images);
            loss.backward();
            optimizer.step();
            train_loss += loss.item()*images.size(0);
        train_loss = train_loss/len(data_loader);
        print('Epoch: {} \t Train_loss: {:.6f}'.format(epoch, train_loss));
        

def project(im, network, num_epochs = 200, learning_rate = 0.1):
    im = torch.from_numpy(MNIST.VectorToImage(im)).float();
    inp = im.clone();
    inp.requires_grad_(True);
    
    optimizer = torch.optim.Adam([inp], lr=learning_rate);

    for i in range(1, num_epochs + 1):
        optimizer.zero_grad();
        out = network(inp.unsqueeze(0).unsqueeze(0));
        loss = torch.norm(im-out);
        loss.backward();
        optimizer.step();
        
    return MNIST.ImageToVector(network(inp.unsqueeze(0).unsqueeze(0)).detach().numpy());

def save_network(network, filename):
    torch.save(network.state_dict(), filename);
    
def load_network(network, filename):
    return network.load_state_dict(torch.load(filename));