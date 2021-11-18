import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import os
import random
import numpy as np
import re
import time

class Net(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class StandardDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        unique_id, feature, label = self.data[idx]
        feature = np.float32(feature)
        return (torch.FloatTensor(feature),
                label)


class SISA:
    def __init__(self, full_data, n_shards, n_slices, model, n_classes):
        """
        full_data: list of n sample <unique_id, sample_feature, sample_label>
        n_shards: integer, number of shards
        n_slices: integer, number of slices
        model: string, architecture to use
        keep_indices: list of k <= n sample of unique ids, use for filter unlearned samples
        """
        self.full_data = full_data
        self.n_shards = n_shards
        self.n_slices = n_slices
        self.n_classes = n_classes
        self.model = model
        self.n_samples = len(full_data)
        self.sample2ss = []
        for i in range(self.n_samples):
            current_shard = random.sample(list(range(self.n_shards)), 1)[0]
            current_slice = random.sample(list(range(self.n_slices)), 1)[0]
            self.sample2ss.append([current_shard, current_slice])
        self.models = [[None for j in range(self.n_slices)]
                       for i in range(self.n_shards)]

        # training configurations
        self.batch_size = 16
        self.epochs = 10

    def _update(self, remove_ids):
        # update self.sample2shard, remove_indices -> move to a new class
        # update which shard to be re-trained, including slice step-point
        # compatible to run before _train for unlearning task
        unlearn_shard = self.n_shards + 1  # this shard does not exist before
        removed_ss = []
        for i in range(self.n_samples):
            unique_id, feature, label = self.full_data[i]
            if(unique_id in remove_ids):
                removed_ss.append(self.sample2ss[i])
                self.sample2ss[i] = [unlearn_shard, 0]

        # figure out which shard to be re-trained / from which slice
        retrain_shards = [False for _ in range(self.n_shards)]
        retrain_slices = [self.n_slices-1 for _ in range(self.n_shards)]
        for (retrain_shard, retrain_slice) in removed_ss:
            retrain_shards[retrain_shard] = True
            retrain_slices[retrain_shard] = min(
                retrain_slice, retrain_slices[retrain_shard])
        return retrain_shards, retrain_slices

    def _unlearn(self, retrain_shards, retrain_slices, save_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in range(self.n_shards):
            if(retrain_shards[i]):
                starting_slice = retrain_slices[i]
                for j in range(starting_slice, self.n_slices):
                    self._train(i, j, self.batch_size, self.epochs,
                                save_path=save_path, device=device)

    def unlearn_do_all(self, remove_ids, save_path='./results_unlearned'):
        retrain_shards, retrain_slices = self._update(remove_ids)
        print("Retrain Shards: ", retrain_shards)
        print("Retrain Slices: ", retrain_slices)
        os.makedirs(save_path, exist_ok=True)
        self._unlearn(retrain_shards, retrain_slices, save_path)
        print('Finish unlearning ...')

    def learn_do_all(self, save_path='./results'):
        os.makedirs(save_path, exist_ok=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in range(self.n_shards):
            for j in range(self.n_slices):
                self._train(i, j, self.batch_size, self.epochs,
                            save_path=save_path, device=device)
        print('Finish learning ...')

    def _train(self, shard_num, slice_num, batch_size, epochs, device="cpu", save_path="results/", verbose=False):
        # shard_num: integer, [0..n_shard-1]
        # slice_num: integer, [0..n_slice-1]
        tik = time.time()

        # step 1: collect data given a shard and slice, put to dataloader
        feature_dim = len(self.full_data[0][1])
        data = []
        for i in range(self.n_samples):
            current_shard, current_slice = self.sample2ss[i]
            if(current_shard == shard_num and current_slice == slice_num):
                data.append(self.full_data[i])
        dataset = StandardDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # step 2: train model
        if(slice_num == 0 or not self.models[shard_num][slice_num-1]):
            # intialize a new model
            model = torchvision.models.resnet18()
        else:
            # use previous slice ckpt
            model = self.models[shard_num][slice_num-1]
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        current_step = 0
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if(verbose):
                    # print statistics to make sure loss goes down
                    running_loss += loss.item()
                    current_step += 1
                    if (current_step % 200 == 0):
                        print('[%d, %5d] loss: %.3f' %
                              (epoch, current_step, running_loss / 200))
                        running_loss = 0.0

        # step 3: saving the model
        outname = "sh"+str(shard_num)+"sl"+str(slice_num)+".pt"
        PATH = os.path.join(save_path, outname)
        torch.save(model.state_dict(), PATH)
        # put model to SISA models
        self.models[shard_num][slice_num] = model
        print("["+ device + "] Finish training ... Shard: " +
              str(shard_num) + " Slice: " + str(slice_num) + " Using: " + str(time.time() - tik))


class SISA_inference:
    """
    this class do the "Aggregation" in SISA
    majority voting and return accuracy
    """

    def __init__(self, test_data, n_shards, n_slices, model, n_classes, learning_path, unlearning_path=None):
        """
        full_data: list of n sample <unique_id, sample_feature, sample_label>
        if unlearning_path is None, inference before unlearning
        if unlearning path is None, inference after unlearning
        """
        self.test_data = test_data
        self.n_shards = n_shards
        self.n_slices = n_slices
        self.n_classes = n_classes
        self.model = model
        self.n_samples = len(test_data)
        self.feature_dim = len(self.test_data[0][1])
        self.batch_size = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # loading models' checkpoints
        self.models = [None for _ in range(self.n_shards)]
        if(unlearning_path):
            for model_name in os.listdir(unlearning_path):
                current_shard, current_slice = re.findall(r'\d+', model_name)
                # if not the last slice, skip the model
                if(int(current_shard) >= self.n_shards or int(current_slice) < self.n_slices - 1):
                    continue
                saved_path = os.path.join(unlearning_path, model_name)
                model = eval(self.model)(self.feature_dim, self.n_classes)
                model.load_state_dict(torch.load(saved_path))
                model.to(self.device)
                model.eval()
                self.models[int(current_shard)] = model

        # load learning models
        for model_name in os.listdir(learning_path):
            current_shard, current_slice = re.findall(r'\d+', model_name)
            # loaded unlearned models
            if(int(current_shard) >= self.n_shards or self.models[int(current_shard)]):
                continue
            # if not the last slice, skip the model
            if(int(current_slice) < self.n_slices - 1):
                continue
            saved_path = os.path.join(learning_path, model_name)
            model = eval(self.model)(self.feature_dim, self.n_classes)
            model.load_state_dict(torch.load(saved_path))
            model.to(self.device)
            model.eval()
            self.models[int(current_shard)] = model

        # assert that all models are loaded
        assert None not in self.models
        print("All contituent models loaded ...")

    def inference(self):
        # create dataloader
        dataset = StandardDataset(self.test_data)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)

        true_labels, predicted_labels = [], []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                B = inputs.size(0)
                outputs = torch.zeros((B, self.n_classes), device=self.device)
                for j in range(self.n_shards):
                    outputs += self.models[j](inputs)
                y_pred = list(outputs.argmax(dim=-1).detach().cpu().numpy())
                y_true = list(labels.detach().cpu().numpy())

                predicted_labels += y_pred
                true_labels += y_true
        return true_labels, predicted_labels
