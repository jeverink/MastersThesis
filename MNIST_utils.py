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
    train_data = torchvision.datasets.MNIST(root='data', train=True, download = True, transform = transform);
    test_data = torchvision.datasets.MNIST(root='data', train=False, download = True, transform = transform);

    num_workers = 0;
    batch_size = 32;
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers = num_workers);
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers = num_workers);

    return (train_data, test_data), (train_loader, test_loader);
    
    
def getTrainLoader(label, transform = None):
    if transform == None:
        transform = torchvision.transforms.ToTensor();
    train_data = torchvision.datasets.MNIST(root='data', train=True, download = True, transform = transform);
    ids = train_data.targets == label;
    train_data.targets = train_data.targets[ids];
    train_data.data = train_data.data[ids];
    num_workers = 0;
    batch_size = 32;
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers = num_workers);
    return train_loader;

def getImage(loader, num):
    dataiter = iter(loader);
    images, labels = dataiter.next();
    im = images[num][0];
    return im.numpy();

def getImageAsVector(loader, num):
    return getImage(loader, num).flatten();



def ImageToVector(image):
    """
    Converts 2D image to one-dimensional vectors.
    """
    return image.flatten();

def VectorToImage(vector):
    """
    Converts one-dimensional vector to 2D image.
    """
    return vector.reshape((28,28));




def reconstruct_BIP(A, q, x_true, projection, learning_param = 0.0005, iterations = 500, intermediate_accuracy = False):
    """
    Applies binary iterative hard-thesholding/projection.
    """
    
    if intermediate_accuracy:
        acc = (iterations + 1)*[0];
    
    x = projection(random.rand(28*28));
    if intermediate_accuracy:
        acc[0] = linalg.norm(utils.normalize(x)-utils.normalize(x_true));
        
    for i in range(iterations):
        gradient = A.T.dot(q-np.sign(A.dot(x)))
        x = projection(x + learning_param*gradient);
        if intermediate_accuracy:
            acc[i + 1] = linalg.norm(utils.normalize(x)-utils.normalize(x_true));
            
    if not intermediate_accuracy:
        acc = linalg.norm(utils.normalize(x)-utils.normalize(x_true));
        
    return x, acc;


def reconstruct_regularized(A, q, x_true, projection, inv_regularization_param, learning_param = 0.1, iterations = 250, intermediate_accuracy = False):
    """
    Applies sub-gradient projection algorithm for the regularized optimization problem.
    """
    
    if intermediate_accuracy:
        acc = (iterations + 1)*[0];
    

    x = projection(np.random.rand(28*28));
    if intermediate_accuracy:
        acc[0] = linalg.norm(normalize(x)-normalize(x_true));
    for i in range(iterations):
        gradient = x - (1.0/(len(q)*inv_regularization_param))*A.T.dot(q);
        x = projection(x - learning_param*gradient);
        if intermediate_accuracy:
            acc[i + 1] = linalg.norm(normalize(x)-normalize(x_true));
            
    if not intermediate_accuracy:
        acc = linalg.norm(utils.normalize(x)-utils.normalize(x_true));
        
    return x, acc;

def reconstruct_regularized_dithering(A, q, lamb, x_true, projection, learning_param = 0.1, iterations = 400,
                                      intermediate_accuracy = False, intermediate_objective = False):
    """
    Applies sub-gradient projection algorithm for the regularized optimization problem with dithering.
    """
    if intermediate_objective:
        obj = (iterations + 1)*[0];
    if intermediate_accuracy:
        acc = (iterations + 1)*[0];
    
    x = projection(random.rand(28*28));
    
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
