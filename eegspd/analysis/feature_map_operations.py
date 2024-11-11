import torch
from sklearn.metrics import accuracy_score
from pyriemann.utils.distance import pairwise_distance

# collect operation applied on feature maps here


def compute_spectrogram(feature_map, Fs):  #TODO: verify shapes
    batch_size, n_chans, n_samps = feature_map.shape

    fft = torch.fft.rfft(feature_map)
    freqs = torch.fft.rfftfreq(n_samps, d=1/Fs)
    return fft, freqs


def compute_eigs(feature_map, mode='eig'):
    if mode == 'eig':
        s, _ = torch.linalg.eigh(feature_map)
    elif mode == 'svd':
        _, s, _ = feature_map.svd()
    else:
        raise ValueError

    return s


def classify_feature_maps(clf, feature_map_train, feature_map_test, labels_train, labels_test):
    # do log outside of func? -> shouldn't need log if doing MDM, just choose the metric
    # NOTE: flattening of last dim (if needed) should be done before sending to this func

    X_train, y_train = feature_map_train.numpy(), labels_train.numpy()
    X_test, y_test = feature_map_test.numpy(), labels_test.numpy()

    clf.fit(X=X_train, y=y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_true=y_train, y_pred=y_train_pred)
    test_acc = accuracy_score(y_true=y_test, y_pred=y_test_pred)
    return dict(train_acc=train_acc, test_acc=test_acc)


def compute_pairwise_distances(feature_map, metric):
    return pairwise_distance(X=feature_map.numpy(), metric=metric)
