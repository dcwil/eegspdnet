import torch
import torch.nn as nn
import h5py
import numpy as np

from .util import get_feature_maps


def grad_outs_wrt_ins_per_label(outputs, inputs, label_i):
    n_samples = outputs.shape[0]
    ls = []
    for sample_i in range(n_samples):
        sample_grad = torch.autograd.grad(
            outputs=outputs[sample_i, label_i],
            inputs=inputs,
            retain_graph=True
        )[0][sample_i]
        ls.append(sample_grad)
    return torch.stack(ls)


def do_cov_grad_analysis(savedir, model, X, y, labels_dict, softmax=False, cov_layer_name='SCMPool_0'):
    X.requires_grad = True

    logits, feature_maps = get_feature_maps(model, X, no_grad=False, return_logits=True)
    if softmax:
        logits = nn.Softmax()(logits)
        loss = nn.NLLLoss()(logits, y)
        append = 'softmax'
    else:
        loss = nn.CrossEntropyLoss()(logits, y)
        append = 'no_softmax'
    loss.backward(retain_graph=True)

    with h5py.File(savedir.joinpath(f'cov_grads_{append}.h5'), 'w') as h:
        for label_str, label_int in labels_dict.items():
            grads = grad_outs_wrt_ins_per_label(logits, feature_maps[cov_layer_name], label_int)
            grads = grads.numpy().astype(np.float16)  # reduce file size
            h.create_dataset(label_str, data=grads, compression='gzip', compression_opts=9)

