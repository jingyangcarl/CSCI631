import configargparse
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import os
from sklearn.metrics import accuracy_score

from sisa import SISA, SISA_inference

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--basedir', type=str, default='./logs', help='where to store ckpts and logs')

    parser.add_argument('--num_workers', type=int, default=2, help='number of workers of Dataloader')

    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for training and testing')
    parser.add_argument('--model', type=str, default='densenet121', help='model used for training and testing')

    parser.add_argument('--R_requests', type=float, default=0.005, help='ratio of records removing')
    parser.add_argument('--N_shards', type=int, default=10, help='number of shards')
    parser.add_argument('--N_slices', type=int, default=10, help='number of slices')
    parser.add_argument('--N_classes', type=int, default=20, help='number of classes')

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
    data_test, label_test = iter(dataloader_test).next()
    len_train, len_test = data_train.shape[0], data_test.shape[0]
    print(f'Loaded from [{args.dataset}] with batch including training data {data_train.shape} and training label {label_train.shape}')

    # 
    def prepare_data(data, label):
        data_out = []
        ids = list(range(data.shape[0]))
        for i in range(data.shape[0]):
            data_out.append([ids[i], data[i], label[i]])
        return data_out, ids
    data_train, id_train = prepare_data(data_train, label_train)
    data_test, id_test = prepare_data(data_test, label_test)

    # load model
    print('----- Loading Model -----')
    sisa = SISA(data_train, args.N_shards, args.N_slices, args.model, args.N_classes)
    baseline = SISA(data_train, 1, 1, args.model, args.N_classes)
    print(f'Loaded [{args.model}]')
    
    # remove ids
    print('----- Build Removing Records -----')
    # higher slice -> higher probability to get picked
    # here, we use the exponential distribution w.r.t the slice_number
    N_requests = int(len_train*args.R_requests) #0.5% of training data
    distribution = np.zeros((len_train,1))
    for i in range(len_train):
        current_slice = sisa.sample2ss[i][1]
        distribution[i] = np.exp(current_slice)
    distribution /= distribution.sum() # make sure the probability sum up to 1

    remove_ids = []
    for i in range(N_requests):
        sampled_remove_id = np.random.choice(id_train, p=list(distribution.reshape(-1,)), replace=False)
        remove_ids.append(sampled_remove_id)
    print(f'Remove {len(remove_ids)} ids: {remove_ids}')

    # learn on sisa
    print('----- Training on SISA -----')
    expname = f'{args.model}_{args.dataset}'
    path_sisalearn = os.path.join(args.basedir, expname, 'sisa_learn')
    path_sisaunlearn = os.path.join(args.basedir, expname, 'sisa_unlearn')
    sisa.learn_do_all(save_path=path_sisalearn)
    sisa.unlearn_do_all(remove_ids, save_path=path_sisaunlearn)

    # learn on baseline
    print('----- Training on Baseline -----')
    path_baselinelearn = os.path.join(args.basedir, expname, 'baseline_learn')
    path_baselineunlearn = os.path.join(args.basedir, expname, 'baseline_unlearn')
    baseline.learn_do_all(save_path=path_baselinelearn)
    baseline.unlearn_do_all(remove_ids, save_path=path_baselineunlearn)

    # prediction on the original trained models (no unlearning done)
    sisa_inference = SISA_inference(test_data=data_test,
                                    n_shards=args.N_shards,
                                    n_slices=args.N_slices,
                                    model=args.model,
                                    n_classes=args.N_classes,
                                    learning_path=path_sisalearn)
    y_true, y_pred = sisa_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    # prediction on the original unlearned models (no unlearning done)
    sisa_inference = SISA_inference(test_data=data_test,
                                    n_shards=args.N_shards,
                                    n_slices=args.N_slices,
                                    model=args.model,
                                    n_classes=args.N_classes,
                                    learning_path=path_sisalearn,
                                    unlearning_path=path_sisaunlearn)
    y_true, y_pred = sisa_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    # prediction on the original trained models (no unlearning done)
    baseline_inference = SISA_inference(test_data=data_test,
                                             n_shards=1,
                                             n_slices=1,
                                             model=args.model,
                                             n_classes=args.N_classes,
                                             learning_path=path_baselinelearn)
    y_true, y_pred = baseline_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    # prediction on the original unlearned models (no unlearning done)
    baseline_inference = SISA_inference(test_data=data_test,
                                             n_shards=1,
                                             n_slices=1,
                                             model=args.model,
                                             n_classes=args.N_classes,
                                             learning_path=path_baselinelearn,
                                             unlearning_path=path_baselineunlearn)
    y_true, y_pred = baseline_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))