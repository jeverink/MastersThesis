import torch;
from torch.utils.data import DataLoader;
import torchvision;

import numpy as np;
import numpy.random as random;
import numpy.linalg as linalg;

import general_utils as utils;



def load_dataset(transform = None):
    if transform == None:
        transform = torchvision.transforms.ToTensor();
    train_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform);
    test_data = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform);

    num_workers = 2;
    batch_size = 16;
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers);
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers);

    return (train_data, test_data), (train_loader, test_loader);

def getImage(data_loader, num):
    dataiter = iter(data_loader);
    images, labels = dataiter.next();
    im = images[num];
    return im;

def getImageAsVector(data_loader, num):
    return getImage(data_loader, num).numpy().flatten();



def ImageToVector(image):
    return image.flatten();

def VectorToImage(vector):
    return vector.reshape((3,32,32));




def reconstruct_regularized_dithering(A, q, lamb, x_true, projection, learning_param = 0.1, iterations = 400,
                                      intermediate_accuracy = False, intermediate_objective = False):
    """
    Applies sub-gradient projection algorithm for the regularized optimization problem with dithering.
    """
    if intermediate_objective:
        obj = (iterations + 1)*[0];
    if intermediate_accuracy:
        acc = (iterations + 1)*[0];
    
    x = projection(random.rand(3*32*32));
    
    if intermediate_objective:
        obj[0] = (1/2)*linalg.norm(x)*linalg.norm(x)-(lamb/len(q))*(q.dot(A.dot(x)));
    if intermediate_accuracy:
        acc[0] = linalg.norm(x-x_true);
        
    for i in range(iterations):
        gradient = x - (lamb/len(q))*A.T.dot(q);
        x = projection(x - learning_param*gradient);
        
        if intermediate_accuracy:
            acc[i + 1] = linalg.norm(x-x_true);
        if intermediate_objective:
            obj[i + 1] = (1/2)*linalg.norm(x)*linalg.norm(x)-(lamb/len(q))*(q.dot(A.dot(x)));
            
    if not intermediate_accuracy:
        acc = linalg.norm(x-x_true);
    if not intermediate_objective:
        obj = (1/2)*linalg.norm(x)*linalg.norm(x)-(lamb/len(q))*(q.dot(A.dot(x)));
        
    return x, acc, obj;
