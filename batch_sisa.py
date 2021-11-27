from run_sisa import run_sisa
import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--basedir', type=str, default='./logs', help='where to store ckpts and logs')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for training and testing')
    parser.add_argument('--model', type=str, default='vgg11', help='model used for training and testing')

    parser.add_argument('--num_workers', type=int, default=2, help='number of workers of Dataloader')
    parser.add_argument('--R_requests', type=float, default=0.005, help='ratio of records removing')
    parser.add_argument('--N_shards', type=int, default=10, help='number of shards')
    parser.add_argument('--N_slices', type=int, default=10, help='number of slices')
    parser.add_argument('--N_classes', type=int, default=47, help='number of classes')

    return parser

if __name__ == '__main__':

    
    parser = config_parser()
    args = parser.parse_args()

    # run_sisa(args)
    # exit()

    datasets = [
        # 'mnist',
        # 'cifar10',
        'emnist',
    ]

    models = [
        # 'vgg11',
        # 'vgg11_bn',
        # 'vgg13',
        # 'vgg13_bn',
        # 'vgg16',
        # 'vgg16_bn',
        # 'vgg19',
        # 'vgg19_bn',
        'resnet18',
        # 'resnet34',
        # 'resnet50',
        # 'resnet101',
        # 'resnet152',
        # 'densenet121',
        # 'densenet161',
        # 'densenet169',
        # 'densenet201',
    ]

    for dataset in datasets:
        for model in models:
            args.basedir = '/mount/Users/jyang/logs/SISA'
            args.dataset = dataset
            args.model = model
            run_sisa(args)