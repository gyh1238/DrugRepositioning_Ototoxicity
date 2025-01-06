#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import Tensor
from typing import Optional

class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.abs()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)


def build_class_priors(
        labels: Tensor,
        class_priors: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        positive_pseudocount: float = 1.,
        negative_pseudocount: float = 1.,
    ) -> Tensor:
    """build class priors, if necessary.
    For each class, the class priors are estimated as
    (P + sum_i w_i y_i) / (P + N + sum_i w_i),
    where y_i is the ith label, w_i is the ith weight, P is a pseudo-count of
    positive labels, and N is a pseudo-count of negative labels.
    Args:
        labels: A `Tensor` with shape [batch_size, num_classes].
            Entries should be in [0, 1].
        class_priors: None, or a floating point `Tensor` of shape [C]
            containing the prior probability of each class (i.e. the fraction of the
            training data consisting of positive examples). If None, the class
            priors are computed from `targets` with a moving average.
        weights: `Tensor` of shape broadcastable to labels, [N, 1] or [N, C],
            where `N = batch_size`, C = num_classes`
        positive_pseudocount: Number of positive labels used to initialize the class
            priors.
        negative_pseudocount: Number of negative labels used to initialize the class
            priors.
    Returns:
        class_priors: A Tensor of shape [num_classes] consisting of the
          weighted class priors, after updating with moving average ops if created.
    """
    if class_priors is not None:
        return class_priors

    N, C = labels.size()

    weighted_label_counts = (weights * labels).sum(0)

    weight_sum = weights.sum(0)

    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors

def weighted_sigmoid_cross_entropy_with_logits(logits: Tensor,
                                               labels: Tensor,
                                               positive_weights: float = 1.,
                                               negative_weights: float = 1.) -> Tensor:
    """Computes a weighting of sigmoid cross entropy given `logits`.

    Measures the weighted probability error in discrete classification tasks in
    which classes are independent and not mutually exclusive.  For instance, one
    could perform multilabel classification where a picture can contain both an
    elephant and a dog at the same time. The class weight multiplies the
    different types of errors.
    For brevity, let `x = logits`, `z = labels`, `c = positive_weights`,
    `d = negative_weights`  The
    weighed logistic loss is

    ```
    c * z * -log(sigmoid(x)) + d * (1 - z) * -log(1 - sigmoid(x))
    = c * z * -log(1 / (1 + exp(-x))) - d * (1 - z) * log(exp(-x) / (1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
    = c * z * log(1 + exp(-x)) + d * (1 - z) * (x + log(1 + exp(-x)))
    = (1 - z) * x * d + (1 - z + c * z ) * log(1 + exp(-x))
    =  - d * x * z + d * x + (d - d * z + c * z ) * log(1 + exp(-x))
    ```

    To ensure stability and avoid overflow, the implementation uses the identity
    log(1 + exp(-x)) = max(0,-x) + log(1 + exp(-abs(x)))
    and the result is computed as

    ```
    = -d * x * z + d * x
    + (d - d * z + c * z ) * (max(0,-x) + log(1 + exp(-abs(x))))
    ```

    Note that the loss is NOT an upper bound on the 0-1 loss, unless it is divided
    by log(2).

    Args:
        labels: A `Tensor` of type `float32` or `float64`. `labels` can be a 2D
            tensor with shape [batch_size, num_labels] or a 3D tensor with shape
            [batch_size, num_labels, K].
        logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
            shape [batch_size, num_labels, K], the loss is computed separately on each
            slice [:, :, k] of `logits`.
        positive_weights: A `Tensor` that holds positive weights and has the
            following semantics according to its shape:
            scalar - A global positive weight.
            1D tensor - must be of size K, a weight for each 'attempt'
            2D tensor - of size [num_labels, K'] where K' is either K or 1.
            The `positive_weights` will be expanded to the left to match the
            dimensions of logits and labels.
        negative_weights: A `Tensor` that holds positive weight and has the
            semantics identical to positive_weights.

    Returns:
        A `Tensor` of the same shape as `logits` with the componentwise
        weighted logistic losses.
    """
    softplus_term = torch.clamp(-logits, min=0.) + torch.log(1.0 + torch.exp(-torch.abs(logits)))
    weight_dependent_factor = (negative_weights + (positive_weights - negative_weights) * labels)
    return (negative_weights * (logits - labels * logits) + weight_dependent_factor * softplus_term)


def weighted_hinge_loss(logits: Tensor,
                        labels: Tensor,
                        positive_weights: float = 1.,
                        negative_weights: float = 1.) -> Tensor:
    """Computes weighted hinge loss given logits `logits`.

    The loss applies to multi-label classification tasks where labels are
    independent and not mutually exclusive. See also
    `weighted_sigmoid_cross_entropy_with_logits`.

    Args:
        labels: A `Tensor` of type `float32` or `float64`. Each entry must be
            either 0 or 1. `labels` can be a 2D tensor with shape
            [batch_size, num_labels] or a 3D tensor with shape
            [batch_size, num_labels, K].
        logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
            shape [batch_size, num_labels, K], the loss is computed separately on each
            slice [:, :, k] of `logits`.
        positive_weights: A `Tensor` that holds positive weights and has the
            following semantics according to its shape:
            scalar - A global positive weight.
            1D tensor - must be of size K, a weight for each 'attempt'
            2D tensor - of size [num_labels, K'] where K' is either K or 1.
            The `positive_weights` will be expanded to the left to match the
            dimensions of logits and labels.
        negative_weights: A `Tensor` that holds positive weight and has the
            semantics identical to positive_weights.

    Returns:
        A `Tensor` of the same shape as `logits` with the componentwise
        weighted hinge loss.
    """
    positives_term = positive_weights * labels * torch.clamp(1.0 - logits, min=0.)
    negatives_term = (negative_weights * (1.0 - labels) * torch.clamp(1.0 + logits, min=0.))
    return positives_term + negatives_term


def weighted_surrogate_loss(logits: Tensor,
                            labels: Tensor,
                            surrogate_type: str = 'xent',
                            positive_weights: float = 1.,
                            negative_weights: float = 1.) -> Tensor:
    """Returns either weighted cross-entropy or hinge loss.

    For example `surrogate_type` is 'xent' returns the weighted cross
    entropy loss.

    Args:
        labels: A `Tensor` of type `float32` or `float64`. Each entry must be
            between 0 and 1. `labels` can be a 2D tensor with shape
            [batch_size, num_labels] or a 3D tensor with shape
            [batch_size, num_labels, K].
        logits: A `Tensor` of the same type and shape as `labels`. If `logits` has
            shape [batch_size, num_labels, K], each slice [:, :, k] represents an
            'attempt' to predict `labels` and the loss is computed per slice.
            surrogate_type: A string that determines which loss to return, supports
            'xent' for cross-entropy and 'hinge' for hinge loss.
        positive_weights: A `Tensor` that holds positive weights and has the
            following semantics according to its shape:
            scalar - A global positive weight.
            1D tensor - must be of size K, a weight for each 'attempt'
            2D tensor - of size [num_labels, K'] where K' is either K or 1.
            The `positive_weights` will be expanded to the left to match the
            dimensions of logits and labels.
        negative_weights: A `Tensor` that holds positive weight and has the
          semantics identical to positive_weights.

    Returns:
        The weigthed loss.

    Raises:
        ValueError: If value of `surrogate_type` is not supported.
    """
    if surrogate_type == 'xent':
        return weighted_sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
            positive_weights=positive_weights,
            negative_weights=negative_weights)
    elif surrogate_type == 'hinge':
        return weighted_hinge_loss(
            logits=logits,
            labels=labels,
            positive_weights=positive_weights,
            negative_weights=negative_weights)
    raise ValueError('surrogate_type %s not supported.' % surrogate_type)
