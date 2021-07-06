"""
This file contains general functions regarding:
File managment, reproducibility and measurements.
"""

import numpy as np;
import numpy.linalg as linalg;
import numpy.random as random;

import pickle;

import torch;
import random as rnd;
    
    
# File managment 
def saveData(data, filename):
    """
    Saves data to file using pickle.
    """
    pickle.dump(data, open(filename, "wb"));
    
def loadData(filename):
    """
    Loades data from file using pickle.
    """
    return pickle.load(open(filename, "rb"));

# Reproducibility
def reset_seeds():
    """
    Resets seeds of random number generators.
    """
    torch.manual_seed(0);
    rnd.seed(0);
    random.seed(0);


# Measurements
def normalize(v):
    """
    Normalizes vector.
    """
    norm = linalg.norm(v);
    if norm == 0:
        return v;
    return v/norm;

def normelized_error(x_true, x):
    return linalg.norm(normalize(x_true) - normalize(x));

def quantize(unq_measurements):
    """
    Quantizes measurements to one-bit.
    """
    return np.sign(unq_measurements);

def generateMeasurements_Gaussian(inp, m, sigma = 0.0):
    """
    Generates random Gaussian measurements with standard Gaussian measurement vectors
    and Guassian noise with specified standard deviation.
    """
    A = random.randn(m, inp.shape[0]);
    q = (A @ inp) + sigma*random.randn(m);
    return A, q;

def generateDithering(m, lamb):
    """
    Generates uniform dithering between lambda and -lambda.
    """
    return lamb*(random.rand(m)*2 - 1);

def generateMeasurements_Gaussian_dithering(image_vec, m, sigma, lamb):
    """
    Generate measurements with Gaussian measurement vectors and Gaussian noise.
    """
    A = random.randn(m, image_vec.shape[0]);
    q = quantize((A @ image_vec) + sigma*random.randn(m) + generateDithering(m, lamb));
    return A, q;

def generateMeasurements_StudentT_dithering(image_vec, m, sigma, dof, lamb):
    """
    Generate measurements with Student-t measurement vectors and Gaussian noise.
    """
    A = np.sqrt((dof-2.0)/(dof))*random.standard_t(dof, (m,image_vec.shape[0]));
    q = quantize((A @ image_vec) + sigma*random.randn(m) + generateDithering(m, lamb));
    return A, q;

def generateMeasurements_Laplace_dithering(image_vec, m, sigma, lamb):
    """
    Generate measurements with Laplace/double exponential measurement vectors and Gaussian noise.
    """
    A = random.laplace(scale = 0.5*np.sqrt(2), size = (m,image_vec.shape[0]));
    q = quantize((A @ image_vec) + sigma*random.randn(m) + generateDithering(m, lamb));
    return A, q;
