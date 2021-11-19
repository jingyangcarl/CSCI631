import configargparse
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from sisa import SISA, SISA_inference

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--basedir', type=str, default='./logs', help='where to store ckpts and logs')
    parser.add_argument('--expname', type=str, default='./test', help='experiment name')

    parser.add_argument('--num_workers', type=int, default=2, help='number of workers of Dataloader')

    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for training and testing')
    parser.add_argument('--model', type=str, default='fc2', help='model used for training and testing')

    parser.add_argument('--N_shards', type=int, default=10, help='number of shards')
    parser.add_argument('--N_slices', type=int, default=10, help='number of slices')
    parser.add_argument('--N_classes', type=int, default=10, help='number of classes')

    return parser

def load_data(args):

    if args.dataset == 'mnist':
        data_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
        data_test = datasets.MNIST(root='data', train=False, transform=ToTensor())
    elif args.dataset == 'cifar10':
        data_train = datasets.CIFAR10(root='data', train=True, transform=ToTensor(), download=True)
        data_test = datasets.CIFAR10(root='data', train=False, transform=ToTensor())

    try:
        dataloader_train = DataLoader(data_train, batch_size=len(data_train), shuffle=True, num_workers=args.num_workers)
        dataloader_test = DataLoader(data_test, batch_size=len(data_test), shuffle=False, num_workers=args.num_workers)
    except NameError:
        print(f'{args.dataset} is not valid')
    
    return dataloader_train, dataloader_test

def load_model(args):

    if args.model == 'resnet18':
        model = torchvision.models.resnet18
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50
    elif args.model == 'resnet101':
        model = torchvision.models.resnet101
    
    return model

if __name__ == '__main__':
    
    # load config
    print('----- Parsing Configuration -----')
    parser = config_parser()
    args = parser.parse_args()

    # load data
    print('----- Loading Data -----')
    dataloader_train, dataloader_test = load_data(args)
    data_train, label_train = iter(dataloader_train).next()
    print(f'Loading from [{args.dataset}] with batch including training data {data_train.shape} and training label {label_train.shape}')

    # load model
    print('----- Loading Model -----')
    
    sisa = SISA(data_train.numpy(), args.N_shards, args.N_slices, args.model, args.N_classes)


    # prepare environment
    print('----- Setting Environment -----')


