import collections
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize


# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
CORRECT_CONF = 'correct_conf'
INCORRECT_CONF = 'incorrect_conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=15):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][CORRECT_CONF] = 0
        bin_dict[i][INCORRECT_CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=15):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]

        if math.isnan(num_bins * confidence):
            print(confs)

        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][CORRECT_CONF] = bin_dict[binn][CORRECT_CONF] + (confidence if (label == prediction) else 0)
        bin_dict[binn][INCORRECT_CONF] = bin_dict[binn][INCORRECT_CONF] + (0 if (label == prediction) else confidence)
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    bin_stats_dict = collections.defaultdict(dict)
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * abs(bin_accuracy - bin_confidence)

        # Save the bin stats to be returned
        bin_lower, bin_upper = i*(1/num_bins), (i+1)*(1/num_bins)
        bin_stats_dict[i]['lower_bound'] = bin_lower
        bin_stats_dict[i]['upper_bound'] = bin_upper
        bin_stats_dict[i]['prop_in_bin'] = float(bin_count)/num_samples
        bin_stats_dict[i]['accuracy_in_bin'] = bin_accuracy
        bin_stats_dict[i]['avg_confidence_in_bin'] = bin_confidence
        bin_stats_dict[i]['ece'] = bin_stats_dict[i]['avg_confidence_in_bin'] - bin_stats_dict[i]['accuracy_in_bin']

    return ece, bin_stats_dict


def maximum_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        ce.append(abs(bin_accuracy - bin_confidence))
    return max(ce)


def adaECE_error(confs, preds, labels, num_bins=15):

    def histedges_equalN(x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, num_bins + 1), np.arange(npt), np.sort(x))

    confidences, predictions, labels = torch.FloatTensor(confs), torch.FloatTensor(preds), torch.FloatTensor(labels)

    accuracies = predictions.eq(labels)
    n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach()))
    # print(n,confidences.shape,bin_boundaries)

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1)
    bin_dict = collections.defaultdict(dict)
    bin_num = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            accuracy_in_bin = torch.zeros(1)
            avg_confidence_in_bin = torch.zeros(1)

        # Save the bin stats to be returned
        bin_dict[bin_num]['lower_bound'] = bin_lower
        bin_dict[bin_num]['upper_bound'] = bin_upper
        bin_dict[bin_num]['prop_in_bin'] = prop_in_bin.item()
        bin_dict[bin_num]['accuracy_in_bin'] = accuracy_in_bin.item()
        bin_dict[bin_num]['avg_confidence_in_bin'] = avg_confidence_in_bin.item()
        bin_dict[bin_num]['ece'] = bin_dict[bin_num]['avg_confidence_in_bin'] - bin_dict[bin_num]['accuracy_in_bin']
        bin_num += 1

    return ece.item(), bin_dict


def average_calibration_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    non_empty_bins = 0
    ace = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        if bin_count > 0:
            non_empty_bins += 1
        ace += abs(bin_accuracy - bin_confidence)
    return ace / float(non_empty_bins)


def l2_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    l2_sum = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        l2_sum += (float(bin_count) / num_samples) * \
               (bin_accuracy - bin_confidence)**2
        l2_error = math.sqrt(l2_sum)
    return l2_error


def test_classification_net_logits(logits, labels):
    '''
    This function reports classification accuracy and confusion matrix given logits and labels
    from a model.
    '''
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    softmax = F.softmax(logits, dim=1)
    confidence_vals, predictions = torch.max(softmax, dim=1)
    labels_list.extend(labels.cpu().numpy().tolist())
    predictions_list.extend(predictions.cpu().numpy().tolist())
    confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list



def test_classification_net(model, data_loader, device, num_bins=15, num_labels=None):
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []

    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)

            logits = model(data)
            if i == 0:
                fulldataset_logits = logits
            else:
                fulldataset_logits = torch.cat((fulldataset_logits, logits), dim=0)

            log_softmax = F.log_softmax(logits, dim=1)
            log_confidence_vals, predictions = torch.max(log_softmax, dim=1)
            confidence_vals = log_confidence_vals.exp()

            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
            labels_list.extend(label.cpu().numpy().tolist())

    accuracy = accuracy_score(labels_list, predictions_list)
    
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list, fulldataset_logits


# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ClasswiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce








# ece_kde
import torch
from torch import nn


def get_bandwidth(f, device):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param device: The device type: 'cpu' or 'cuda'

    :return: The bandwidth of the kernel
    """
    bandwidths = torch.cat((torch.logspace(start=-5, end=-1, steps=15), torch.linspace(0.2, 1, steps=5)))
    max_b = -1
    max_l = 0
    n = len(f)
    for b in bandwidths:
        log_kern = get_kernel(f, b, device)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log((n-1)*b)
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b


def get_ece_kde(f, y, bandwidth, p, mc_type, device):
    """
    Calculate an estimate of Lp calibration error.

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param y: The vector containing the labels, shape [num_samples]
    :param bandwidth: The bandwidth of the kernel
    :param p: The p-norm. Typically, p=1 or p=2
    :param mc_type: The type of multiclass calibration: canonical, marginal or top_label
    :param device: The device type: 'cpu' or 'cuda'

    :return: An estimate of Lp calibration error
    """
    check_input(f, bandwidth, mc_type)
    if f.shape[1] == 1:
        return 2 * get_ratio_binary(f, y, bandwidth, p, device)
    else:
        if mc_type == 'canonical':
            return get_ratio_canonical(f, y, bandwidth, p, device)
        elif mc_type == 'marginal':
            return get_ratio_marginal_vect(f, y, bandwidth, p, device)
        elif mc_type == 'top_label':
            return get_ratio_toplabel(f, y, bandwidth, p, device)


def get_ratio_binary(f, y, bandwidth, p, device):
    assert f.shape[1] == 1

    log_kern = get_kernel(f, bandwidth, device)

    return get_kde_for_ece(f, y, log_kern, p)


def get_ratio_canonical(f, y, bandwidth, p, device):
    if f.shape[1] > 60:
        # Slower but more numerically stable implementation for larger number of classes
        return get_ratio_canonical_log(f, y, bandwidth, p, device)

    log_kern = get_kernel(f, bandwidth, device)
    kern = torch.exp(log_kern)

    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    kern_y = torch.matmul(kern, y_onehot)
    den = torch.sum(kern, dim=1)
    # to avoid division by 0
    den = torch.clamp(den, min=1e-10)

    ratio = kern_y / den.unsqueeze(-1)
    ratio = torch.sum(torch.abs(ratio - f)**p, dim=1)

    return torch.mean(ratio)


# Note for training: Make sure there are at least two examples for every class present in the batch, otherwise
# LogsumexpBackward returns nans.
def get_ratio_canonical_log(f, y, bandwidth, p, device='cpu'):
    log_kern = get_kernel(f, bandwidth, device)
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_y = torch.log(y_onehot).to(device)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = 0
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (torch.ones([f.shape[0], 1]).to(device) * log_y[:, k].unsqueeze(0))
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        inner_diff = torch.abs(inner_ratio - f[:, k])**p
        final_ratio += inner_diff

    return torch.mean(final_ratio)


def get_ratio_marginal_vect(f, y, bandwidth, p, device):
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_kern_vect = beta_kernel(f, f, bandwidth).squeeze()
    log_kern_diag = torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)
    # Multiclass case
    log_kern_diag_repeated = f.shape[1] * [log_kern_diag]
    log_kern_diag_repeated = torch.stack(log_kern_diag_repeated, dim=2)
    log_kern_vect = log_kern_vect + log_kern_diag_repeated

    return get_kde_for_ece_vect(f, y_onehot, log_kern_vect, p)


def get_ratio_toplabel(f, y, bandwidth, p, device):
    f_max, indices = torch.max(f, 1)
    f_max = f_max.unsqueeze(-1)
    y_max = (y == indices).to(torch.int)

    return get_ratio_binary(f_max, y_max, bandwidth, p, device)


def get_kde_for_ece_vect(f, y, log_kern, p):
    log_kern_y = log_kern * y
    # Trick: -inf instead of 0 in log space
    log_kern_y[log_kern_y == 0] = torch.finfo(torch.float).min

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f)**p

    return torch.sum(torch.mean(ratio, dim=0))


def get_kde_for_ece(f, y, log_kern, p):
    f = f.squeeze()
    N = len(f)
    # Select the entries where y = 1
    idx = torch.where(y == 1)[0]
    if not idx.numel():
        return torch.sum((torch.abs(-f))**p) / N

    if idx.numel() == 1:
        # because of -inf in the vector
        log_kern = torch.cat((log_kern[:idx], log_kern[idx+1:]))
        f_one = f[idx]
        f = torch.cat((f[:idx], f[idx+1:]))

    log_kern_y = torch.index_select(log_kern, 1, idx)

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f)**p

    if idx.numel() == 1:
        return (ratio.sum() + f_one ** p)/N

    return torch.mean(ratio)


def get_kernel(f, bandwidth, device):
    # if num_classes == 1
    if f.shape[1] == 1:
        log_kern = beta_kernel(f, f, bandwidth).squeeze()
    else:
        log_kern = dirichlet_kernel(f, bandwidth).squeeze()
    # Trick: -inf on the diagonal
    return log_kern.to(device) + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)


def beta_kernel(z, zi, bandwidth=0.1):
    p = zi / bandwidth + 1
    q = (1-zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p-1) * torch.log(z) + (q-1) * torch.log(1-z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf


def dirichlet_kernel(z, bandwidth=0.1):
    alphas = z / bandwidth + 1

    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(z), (alphas-1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf


def check_input(f, bandwidth, mc_type):
    assert not isnan(f)
    assert len(f.shape) == 2
    assert bandwidth > 0
    assert torch.min(f) >= 0
    assert torch.max(f) <= 1


def isnan(a):
    return torch.any(torch.isnan(a))






# RBS
from scipy.special import softmax
def BS(logits, labels):
    logits = logits.cpu().numpy()
    labels = labels.cpu().numpy()
    p = softmax(logits, axis=1)
    y = label_binarize(np.array(labels), classes=range(logits.shape[1]))
    return np.average(np.sum((p - y)**2, axis=1))