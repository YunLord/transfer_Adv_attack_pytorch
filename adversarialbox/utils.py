import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler
from skimage import io
from os.path import join
import csv
import os

def truncated_normal(mean=0.0, stddev=1.0, m=1):
    '''
    The generated values follow a normal distribution with specified 
    mean and standard deviation, except that values whose magnitude is 
    more than 2 standard deviations from the mean are dropped and 
    re-picked. Returns a vector of length m
    '''
    samples = []
    for i in range(m):
        while True:
            sample = np.random.normal(mean, stddev)
            if np.abs(sample) <= 2 * stddev:
                break
        samples.append(sample)
    assert len(samples) == m, "something wrong"
    if m == 1:
        return samples[0]
    else:
        return np.array(samples)


# --- PyTorch helpers ---

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(to_var(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


def test(model, loader, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


def attack_over_test_data(model, adversary, params,loader_test,oracle=None):
    """
    Given target model computes accuracy on perturbed data
    """
    total_correct = 0
    total_samples = len(loader_test.dataset)

    # For black-box
    if oracle is not None:
        total_samples -= params['hold_out_size']

    for t, (X, y) in enumerate(loader_test):
        y_pred = pred_batch(X, model)
        X_adv = adversary.perturb(X.numpy(), y_pred)
        X_adv = torch.from_numpy(X_adv)
        if oracle is not None:
            y_pred_adv = pred_batch(X_adv, oracle)
        else:
            y_pred_adv = pred_batch(X_adv, model)
        total_correct += (y_pred_adv.numpy() == y.numpy()).sum()

    acc = total_correct/total_samples

    print('Got %d/%d correct (%.2f%%) on the perturbed data' 
        % (total_correct, total_samples, 100 * acc))

    return acc

def attack_over_test_data_and_save(model, adversary, dataset, args,oracle=None):
    """
    Given target model computes accuracy on perturbed data
    """
    total_correct = 0
    total_samples = len(dataset)

    # For black-box
    if oracle is not None:
        total_samples -= args.hold_out_size
    data_file = []
    save_path=join(args.mode,args.img_save_path)
    if(not os.path.exists(save_path)):
        os.makedirs(save_path)
    for i, (X, y) in enumerate(dataset):
        X=X.unsqueeze(0)
        y_pred = pred_batch(X, model)
        X_adv = adversary.perturb(X.numpy(), y_pred)
        X_adv = torch.from_numpy(X_adv)
        if oracle is not None:
            y_pred_adv = pred_batch(X_adv, oracle)
        else:
            y_pred_adv = pred_batch(X_adv, model)
        img_path = join(save_path , str(i) + '.jpg')
        io.imsave(img_path, X_adv.squeeze(0).squeeze(0).numpy())        
        total_correct += (y_pred_adv.numpy() == y.numpy()).sum()
        data_file.append([img_path,y.numpy(),y_pred_adv.numpy()])
    csv_file=join(save_path,args.img_save_path+'.csv')
    header=['location','label','adv_label']
    with open(csv_file, 'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows([header])
        writer.writerows(data_file)
    acc = total_correct/total_samples

    print('Got %d/%d correct (%.2f%%) on the perturbed data' 
        % (total_correct, total_samples, 100 * acc))

    return acc


def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end
