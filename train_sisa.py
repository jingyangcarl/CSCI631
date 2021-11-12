from sisa import SISA, SISA_inference
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_kddcup99
from sklearn.metrics import accuracy_score
import random

if __name__ == '__main__':

    # Sample dataset
    data_all = fetch_kddcup99(download_if_missing=True)

    features = data_all.data[:, 4:]
    labels = data_all.target
    print(features[0:5], labels[0:5])

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    N, D = features.shape
    unique_ids = list(range(N))
    data = []
    for i in range(N):
        data.append([unique_ids[i], features[i], labels[i]])

    # build sisa
    n_shards, n_slices = 5, 5
    model = "Net"
    n_classes = 23
    sisa = SISA(data, n_shards, n_slices, model, n_classes)

    # learn on sisa
    sisa.learn_do_all()

    # unlearning on SISA
    n_requests = 15
    remove_ids = random.sample(unique_ids, n_requests)
    sisa.unlearn_do_all(remove_ids)

    # prediction on the original trained models (no unlearning done)
    # prediction on the training set (did not split train/test before)
    sisa_inference = SISA_inference(test_data=data,
                                    n_shards=5,
                                    n_slices=5,
                                    model="Net",
                                    n_classes=n_classes,
                                    learning_path="results/")
    y_true, y_pred = sisa_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    # prediction on the original unlearned models (no unlearning done)
    # prediction on the training set (did not split train/test before)
    sisa_inference = SISA_inference(test_data=data,
                                    n_shards=5,
                                    n_slices=5,
                                    model="Net",
                                    n_classes=n_classes,
                                    learning_path="results/",
                                    unlearning_path="results_unlearned/")
    y_true, y_pred = sisa_inference.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))


    # Baseline training and re-training from scratch
    # set num_shards = 1 and num_slices = 1
    n_shards, n_slices = 1, 1
    model = "Net"
    n_classes = 23
    sisa_baseline = SISA(data, n_shards, n_slices, model, n_classes)

    # learning on SISA baseline
    sisa_baseline.learn_do_all()

    #unlearning on SISA baseline, same remove_ids as SISA from above
    sisa_baseline.unlearn_do_all(remove_ids)

    sisa_inference_baseline = SISA_inference(test_data=data,
                                n_shards=1,
                                n_slices=1,
                                model="Net",
                                n_classes=n_classes,
                                learning_path = "results/")
    y_true, y_pred = sisa_inference_baseline.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))

    sisa_inference_baseline = SISA_inference(test_data=data,
                                n_shards=1,
                                n_slices=1,
                                model="Net",
                                n_classes=n_classes,
                                learning_path = "results/",
                                unlearning_path = "results_unlearned/")
    y_true, y_pred = sisa_inference_baseline.inference()
    print("Accuracy Score: ", accuracy_score(y_true, y_pred))