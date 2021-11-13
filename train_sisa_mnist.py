from sisa import SISA, SISA_inference
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_kddcup99
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
    n_shards, n_slices = 5, 10
    model = "Net"
    n_classes = 23
    sisa = SISA(data_train, n_shards, n_slices, model, n_classes)

    # remove ids
    # higher slice -> higher probability to get picked
    # here, we use the exponential distribution w.r.t the slice_number
    n_requests = int(len_train/200) #0.5% of training data
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

    # learn on sisa
    sisa.learn_do_all(save_path='results_sisa')

    # unlearning on SISA
    sisa.unlearn_do_all(remove_ids, save_path='results_ul_sisa')

    # prediction on the original trained models (no unlearning done)
    # prediction on the training set (did not split train/test before)
    sisa_inference = SISA_inference(test_data=data_test,
                                    n_shards=n_shards,
                                    n_slices=n_slices,
                                    model=model,
                                    n_classes=n_classes,
                                    learning_path="results_sisa/")
    y_true, y_pred = sisa_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    # prediction on the original unlearned models (no unlearning done)
    # prediction on the training set (did not split train/test before)
    sisa_inference = SISA_inference(test_data=data_test,
                                    n_shards=n_shards,
                                    n_slices=n_slices,
                                    model=model,
                                    n_classes=n_classes,
                                    learning_path="results_sisa/",
                                    unlearning_path="results_ul_sisa/")
    y_true, y_pred = sisa_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    # Baseline training and re-training from scratch
    # set num_shards = 1 and num_slices = 1
    n_shards, n_slices = 1, 1
    model = "Net"
    n_classes = 23
    baseline = SISA(data_train, n_shards, n_slices, model, n_classes)

    # learning on SISA baseline
    baseline.learn_do_all(save_path='results_baseline')

    # unlearning on SISA baseline, same remove_ids as SISA from above
    baseline.unlearn_do_all(remove_ids, save_path='results_ul_baseline')

    baseline_inference = SISA_inference(test_data=data_test,
                                             n_shards=n_shards,
                                             n_slices=n_slices,
                                             model=model,
                                             n_classes=n_classes,
                                             learning_path="results_baseline/")
    y_true, y_pred = baseline_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    baseline_inference = SISA_inference(test_data=data_test,
                                             n_shards=n_shards,
                                             n_slices=n_slices,
                                             model=model,
                                             n_classes=n_classes,
                                             learning_path="results_baseline/",
                                             unlearning_path="results_ul_baseline/")
    y_true, y_pred = baseline_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))
