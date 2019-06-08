# liyi, 2019/6/8

import torch

import video_prediction.globalvar as gl
device = gl.get_value()

def l1_loss(pred, target):
    criterion = torch.nn.L1Loss()
    return criterion(pred, target)

def l2_loss(pred, target):
    criterion = torch.nn.MSELoss()
    return criterion(pred, target)

def normalize_tensor(tensor, eps=1e-10):
    norm_factor = torch.norm(tensor, dim=-1, keepdim=True)
    return tensor / (norm_factor + eps)


def cosine_distance(tensor0, tensor1, keep_axis=None):
    tensor0 = normalize_tensor(tensor0)
    tensor1 = normalize_tensor(tensor1)
    return torch.mean(torch.pow(tensor0 - tensor1, 2).sum(dim=-1)) / 2.0
                                

def charbonnier_loss(x, epsilon=0.001):
    return torch.mean(torch.sqrt(torch.pow(x, 2) + torch.powe(epsilon, 2)))


def gan_loss(logits, labels, gan_loss_type):
    # use 1.0 (or 1.0 - discrim_label_smooth) for real data and 0.0 for fake data
    if gan_loss_type == 'GAN':
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if labels in (0.0, 1.0):
            # labels = tf.constant(labels, dtype=logits.dtype, shape=logits.get_shape())
            # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
            labels = (labels * torch.ones(logits.shape).cuda(device)).type_as(logits)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = torch.mean(criterion(logits, labels))
        else:
            loss = torch.mean(sigmoid_kl_with_logits(logits, labels))
    elif gan_loss_type == 'LSGAN':
        # discrim_loss = tf.reduce_mean((tf.square(predict_real - 1) + tf.square(predict_fake)))
        # gen_loss = tf.reduce_mean(tf.square(predict_fake - 1))
        loss = torch.mean(torch.pow(logits - labels, 2))
    elif gan_loss_type == 'SNGAN':
        # this is the form of the loss used in the official implementation of the SNGAN paper, but it leads to
        # worse results in our video prediction experiments
        if labels == 0.0:
            #loss =torch.mean(tf.nn.softplus(logits))
            loss = torch.mean(torch.log(torch.exp(logits) + 1))
        elif labels == 1.0:
            loss = torch.mean(torch.log(torch.exp(-logits) + 1))
        else:
            raise NotImplementedError
    else:
        raise ValueError('Unknown GAN loss type %s' % gan_loss_type)
    return loss


def kl_loss(mu, log_sigma_sq, mu2=None, log_sigma2_sq=None):
    if mu2 is None and log_sigma2_sq is None:
        sigma_sq = torch.exp(log_sigma_sq)
        return -0.5 * torch.mean(torch.sum(1 + log_sigma_sq - torch.pow(mu, 2) - sigma_sq, dim=-1))
    else:
        mu1 = mu
        log_sigma1_sq = log_sigma_sq
        return torch.mean(torch.sum(
            (log_sigma2_sq - log_sigma1_sq) / 2
            + (torch.exp(log_sigma1_sq) + torch.pow(mu1 - mu2, 2)) / (2 * torch.exp(log_sigma2_sq))
            - 1 / 2, dim=-1))

def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(logits, torch.ones_like(logits) * targets) - entropy
