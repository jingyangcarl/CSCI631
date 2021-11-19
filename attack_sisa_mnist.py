from sisa import SISA, SISA_completeness
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np

if __name__ == '__main__':

    # Sample dataset
    data_train = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    data_test = datasets.MNIST(root='data', train=False, transform=ToTensor())
    print(f'train: {data_train.data.size()}; test: {data_test.data.size()}')

    # get img and label
    img_train, label_train = data_train.data, data_train.targets # 60000*28*28, 60000
    img_test, label_test = data_test.data, data_test.targets # 10000*28*28, 10000
    len_train, len_test = img_train.shape[0], img_test.shape[0]

    # reshape
    feature_train = img_train.reshape((len_train, -1)) # 60000, 784
    feature_test = img_test.reshape((len_test, -1)) # 10000, 784

    id_train = list(range(len_train))
    id_test = list(range(len_test))

    data_train, data_test = [], []
    for i in range(len_train):
        data_train.append([id_train[i], feature_train[i], label_train[i]])
    for i in range(len_test):
        data_test.append([id_test[i], feature_test[i], label_test[i]])
    print(f'dataset prepared')

    # build sisa
    n_shards, n_slices = 10, 10
    model = "Net"
    n_classes = 10
    sisa = SISA(data_train, n_shards, n_slices, model, n_classes)

    # remove ids
    # higher slice -> higher probability to get picked
    # here, we use the exponential distribution w.r.t the slice_number
    n_requests = int(len_train/10) # 10% of training data
    distribution = np.zeros((len_train,1))
    for i in range(len_train):
        current_slice = sisa.sample2ss[i][1]
        distribution[i] = np.exp(current_slice)
    distribution /= distribution.sum() # make sure the probability sum up to 1
    
    remove_ids = []
    for i in range(n_requests):
        sampled_remove_id = np.random.choice(id_train, p=list(distribution.reshape(-1,)), replace=False)
        remove_ids.append(sampled_remove_id)
    
    #  this remove_ids is for uniform distribution
    #remove_ids = random.sample(id_train, n_requests)
    
    learning_path = 'results_sisa'
    unlearning_path = 'results_ul_sisa'
    # learn on sisa
    sisa.learn_do_all(save_path=learning_path)

    # unlearning on SISA
    sisa.unlearn_do_all(remove_ids, save_path=unlearning_path)
    
    attack_data, attack_labels = [], []
    cnt_0, cnt_1 = 0, 0
    for i in range(len(feature_train)):
        if(i not in remove_ids and cnt_0 < n_requests):
            attack_data.append(feature_train[i].numpy())
            attack_labels.append(0) # seen during training but not deleted
            cnt_0 += 1
        if(i in remove_ids):
            attack_data.append(feature_train[i].numpy())
            attack_labels.append(1) # seen during training but deleted
            
    for i in range(len(feature_test)):
        if(cnt_1 < n_requests):
            attack_data.append(feature_test[i].numpy())
            attack_labels.append(2) # not seen during training
            cnt_1 += 1
            
    attacker = SISA_completeness(data=attack_data,
                                 labels=attack_labels,
                                 n_shards=n_shards,
                                 n_slices=n_slices,
                                 n_classes=n_classes,
                                 model=model,
                                 learning_path=learning_path,
                                 unlearning_path=unlearning_path)
    attacker.attack()


