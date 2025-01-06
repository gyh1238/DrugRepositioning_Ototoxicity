import torch
from torch import Tensor
from typing import Optional, Tuple
from torch.nn.modules.loss import _Loss
from loss_utils import lagrange_multiplier, weighted_surrogate_loss, build_class_priors

from torch.nn.parameter import Parameter

class RecallAtPrecisionLoss(_Loss):
    """Computes recall at precision loss.

    The loss is based on a surrogate of the form
        wt * w(+) * loss(+) + wt * w(-) * loss(-) - c * pi,
    where:
    - w(+) =  1 + lambdas * (1 - target_precision)
    - loss(+) is the cross-entropy loss on the positive examples
    - w(-) = lambdas * target_precision
    - loss(-) is the cross-entropy loss on the negative examples
    - wt is a scalar or tensor of per-example weights
    - c = lambdas * (1 - target_precision)
    - pi is the class_priors.

    The per-example weights change not only the coefficients of individual
    training examples, but how the examples are counted toward the constraint.
    If `class_priors` is given, it MUST take `weights` into account. That is,
        class_priors = P / (P + N)
    where
        P = sum_i (wt_i on positives)
        N = sum_i (wt_i on negatives).

    Args:
    logits: A `Tensor` with the same shape as `classes`.
    classes: A `Tensor` of shape [batch_size] or [batch_size, num_classes].
    target_precision: The precision at which to compute the loss. Can be a
        floating point value between 0 and 1 for a single precision value, or a
        `Tensor` of shape [num_classes], holding each class's target precision
        value.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
        [batch_size] or [batch_size, num_classes].
    dual_rate_factor: A floating point value which controls the step size for
        the Lagrange multipliers.
    class_priors: None, or a floating point `Tensor` of shape [num_classes]
        containing the prior probability of each class (i.e. the fraction of the
        training data consisting of positive examples). If None, the class
        priors are computed from `classes` with a moving average. See the notes
        above regarding the interaction with `weights` and do not set this unless
        you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
        should be used for indicator functions.

    reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

    Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
        loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
        more details, see http://arxiv.org/pdf/1608.04802.pdf.
        lambdas: A Tensor of shape [num_classes] consisting of the Lagrange
            multipliers.
        class_priors: A Tensor of shape [num_classes] consisting of the prior
            probability of each class learned by the loss, if not provided.
            true_positives_lower_bound: Lower bound on the number of true positives
            given `classes` and `logits`. This is the same lower bound which is used
            in the loss expression to be optimized.
        false_positives_upper_bound: Upper bound on the number of false positives
            given `classes` and `logits`. This is the same upper bound which is used
            in the loss expression to be optimized.

    Raises:
        ValueError: If `logits` and `classes` do not have the same shape.
    """
    __constants__ = ['num_classes', 'target_precision', 'dual_rate_factor', 'surrogate_type', 'reduction']
    num_classes: int
    target_precision: float
    dual_rate_factor: float
    lambdas_variable: Tensor

    def __init__(self, num_classes: int, target_precision: float, dual_rate_factor: float = 0.1,
                 surrogate_type: str = 'xent',
                 size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(RecallAtPrecisionLoss, self).__init__(size_average, reduce, reduction)
        self.num_classes = num_classes
        self.target_precision = target_precision
        self.dual_rate_factor = dual_rate_factor
        self.surrogate_type = surrogate_type

        # Create lambdas.
        self.lambdas_variable = Parameter(torch.ones(num_classes))

        if surrogate_type == 'xent':
            self.maybe_log2 = Parameter(torch.full((1,), 2.).log_(), requires_grad=False)
        else:
            self.maybe_log2 = Parameter(torch.ones(1), requires_grad=False)


    def forward(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:

        N, C = logits.size()
        dtype = logits.dtype
        device = logits.device

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        # Maybe create weights.
        if weights is None:
            weights = torch.ones((N, C), dtype=dtype, device=device)

        # Create class priors
        class_priors = build_class_priors(targets, weights=weights)

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [num_classes]
        lambdas = lagrange_multiplier(self.lambdas_variable)

        # Surrogate loss.
        positive_weights = 1.0 + lambdas * (1.0 - self.target_precision)
        negative_weights = lambdas * self.target_precision
        weighted_loss = weights * weighted_surrogate_loss(
            logits=logits,
            labels=targets,
            surrogate_type=self.surrogate_type,
            positive_weights=positive_weights,
            negative_weights=negative_weights)

        lambda_term = lambdas * (1.0 - self.target_precision) * class_priors * self.maybe_log2
        loss = weighted_loss - lambda_term

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class PrecisionAtRecallLoss(_Loss):
    """Computes precision at recall loss.

    The loss is based on a surrogate of the form
        wt * loss(-) + lambdas * (pi * (b - 1) + wt * loss(+))
    where:
    - loss(-) is the cross-entropy loss on the negative examples
    - loss(+) is the cross-entropy loss on the positive examples
    - wt is a scalar or tensor of per-example weights
    - b is the target recall
    - pi is the label_priors.

    The per-example weights change not only the coefficients of individual
    training examples, but how the examples are counted toward the constraint.
    If `label_priors` is given, it MUST take `weights` into account. That is,
      label_priors = P / (P + N)
    where
      P = sum_i (wt_i on positives)
      N = sum_i (wt_i on negatives).

    Args:
    labels: A `Tensor` of shape [batch_size] or [batch_size, num_labels].
    logits: A `Tensor` with the same shape as `labels`.
    target_recall: The recall at which to compute the loss. Can be a floating
      point value between 0 and 1 for a single target recall value, or a
      `Tensor` of shape [num_labels] holding each label's target recall value.
    weights: Coefficients for the loss. Must be a scalar or `Tensor` of shape
      [batch_size] or [batch_size, num_labels].
    dual_rate_factor: A floating point value which controls the step size for
      the Lagrange multipliers.
    label_priors: None, or a floating point `Tensor` of shape [num_labels]
      containing the prior probability of each label (i.e. the fraction of the
      training data consisting of positive examples). If None, the label
      priors are computed from `labels` with a moving average. See the notes
      above regarding the interaction with `weights` and do not set this unless
      you have a good reason to do so.
    surrogate_type: Either 'xent' or 'hinge', specifying which upper bound
      should be used for indicator functions.
    lambdas_initializer: An initializer for the Lagrange multipliers.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for the variables.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    scope: Optional scope for `variable_scope`.

    Returns:
    loss: A `Tensor` of the same shape as `logits` with the component-wise
      loss.
    other_outputs: A dictionary of useful internal quantities for debugging. For
      more details, see http://arxiv.org/pdf/1608.04802.pdf.
      lambdas: A Tensor of shape [num_labels] consisting of the Lagrange
        multipliers.
      label_priors: A Tensor of shape [num_labels] consisting of the prior
        probability of each label learned by the loss, if not provided.
      true_positives_lower_bound: Lower bound on the number of true positives
        given `labels` and `logits`. This is the same lower bound which is used
        in the loss expression to be optimized.
      false_positives_upper_bound: Upper bound on the number of false positives
        given `labels` and `logits`. This is the same upper bound which is used
        in the loss expression to be optimized.
    """

    __constants__ = ['num_classes', 'target_recall', 'dual_rate_factor', 'surrogate_type', 'reduction']
    num_classes: int
    target_recall: float
    dual_rate_factor: float
    lambdas_variable: Tensor

    def __init__(self, num_classes: int, target_recall: float, dual_rate_factor: float = 0.1,
                 surrogate_type: str = 'xent',
                 size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PrecisionAtRecallLoss, self).__init__(size_average, reduce, reduction)
        self.num_classes = num_classes
        self.target_recall = target_recall
        self.dual_rate_factor = dual_rate_factor
        self.surrogate_type = surrogate_type

        # Create lambdas.
        self.lambdas_variable = Parameter(torch.ones(num_classes))

        if surrogate_type == 'xent':
            self.maybe_log2 = Parameter(torch.full((1,), 2.).log_(), requires_grad=False)
        else:
            self.maybe_log2 = Parameter(torch.ones(1), requires_grad=False)


    def forward(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:

        N, C = logits.size()
        dtype = logits.dtype
        device = logits.device

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        # Maybe create weights.
        if weights is None:
            weights = torch.ones((N, C), dtype=dtype, device=device)

        # Create class priors
        class_priors = build_class_priors(targets, weights=weights)

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 1D `Tensor` of shape [num_classes]
        lambdas = lagrange_multiplier(self.lambdas_variable)

        # Surrogate loss.
        positive_weights = lambdas
        negative_weights = 1.
        weighted_loss = weights * weighted_surrogate_loss(
            logits=logits,
            labels=targets,
            surrogate_type=self.surrogate_type,
            positive_weights=positive_weights,
            negative_weights=negative_weights)

        lambda_term = lambdas * class_priors * (self.target_recall - 1.) * self.maybe_log2
        loss = weighted_loss + lambda_term

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class PrecisionRecallAUCLoss(_Loss):

    __constants__ = ['num_classes', 'num_anchors', 'dual_rate_factor', 'surrogate_type', 'reduction']
    num_classes: int
    precision_range: Tuple[float, float]
    num_anchors: int
    dual_rate_factor: float
    lambdas_variable: Tensor

    def __init__(self, num_classes: int, precision_range: Tuple[float, float] = (0., 1.),
                 num_anchors: int = 20, dual_rate_factor: float = 0.1,
                 surrogate_type: str = 'xent',
                 size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PrecisionRecallAUCLoss, self).__init__(size_average, reduce, reduction)
        self.num_classes = num_classes
        self.precision_range = precision_range
        self.num_anchors = num_anchors
        self.dual_rate_factor = dual_rate_factor
        self.surrogate_type = surrogate_type

        # Create Precision values.
        # Validate precision_range.
        if not 0 <= precision_range[0] <= precision_range[-1] <= 1:
            raise ValueError('precision values must obey 0 <= %f <= %f <= 1' %
                             (precision_range[0], precision_range[-1]))
        if not 0 < len(precision_range) < 3:
            raise ValueError('length of precision_range (%d) must be 1 or 2' %
                             len(precision_range))

        # Sets precision_values uniformly between min_precision and max_precision.
        self.precision_values = Parameter(torch.reshape(
            torch.linspace(precision_range[0], precision_range[1], num_anchors + 2)[1:-1],
            (1, 1, num_anchors)
        ), requires_grad=False)
        self.precision_delta = Parameter(self.precision_values[0, 0, 0] - precision_range[0], requires_grad=False)

        # Create lambdas.
        self.lambdas_variable = Parameter(torch.ones([1, num_classes, num_anchors]))

        # Create biases.
        self.biases = Parameter(torch.zeros([1, num_classes, num_anchors]))

        if surrogate_type == 'xent':
            self.maybe_log2 = Parameter(torch.full((1,), 2.).log_(), requires_grad=False)
        else:
            self.maybe_log2 = Parameter(torch.ones(1), requires_grad=False)


    def forward(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:

        N, C = logits.size()
        dtype = logits.dtype
        device = logits.device

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        # Maybe create weights.
        if weights is None:
            weights = torch.ones((N, C), dtype=dtype, device=device)

        # Create class priors
        class_priors = build_class_priors(targets, weights=weights)

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 2D `Tensor` of shape [num_classes, num_anchors]
        lambdas = lagrange_multiplier(self.lambdas_variable)

        # Reshape tensors
        logits = torch.unsqueeze(logits, -1)
        targets = torch.unsqueeze(targets, -1)
        weights = torch.unsqueeze(weights, -1)
        class_priors = torch.reshape(class_priors, (1, C, 1))

        # Surrogate loss.
        positive_weights = 1. + lambdas * (1. -  self.precision_values)
        negative_weights = lambdas * self.precision_values
        weighted_loss = weights * weighted_surrogate_loss(
            logits=logits + self.biases,
            labels=targets,
            surrogate_type=self.surrogate_type,
            positive_weights=positive_weights,
            negative_weights=negative_weights)

        lambda_term = lambdas * class_priors * (1. - self.precision_values) * self.maybe_log2
        per_anchor_loss = weighted_loss - lambda_term
        per_class_loss = self.precision_delta * torch.sum(per_anchor_loss, 2)

        # Normalize the AUC such that a perfect score function will have AUC 1.0.
        # Because precision_range is discretized into num_anchors + 1 intervals
        # but only num_anchors terms are included in the Riemann sum, the
        # effective length of the integration interval is `delta` less than the
        # length of precision_range.
        scaled_loss = torch.div(per_class_loss,
            self.precision_range[1] - self.precision_range[0] - self.precision_delta)

        if self.reduction == 'mean':
            return scaled_loss.mean()
        elif self.reduction == 'sum':
            return scaled_loss.sum()
        return scaled_loss

class PrecisionRecallAUCLoss2(_Loss):

    __constants__ = ['num_classes', 'num_anchors', 'dual_rate_factor', 'surrogate_type', 'reduction']
    num_classes: int
    recall_range: Tuple[float, float]
    num_anchors: int
    dual_rate_factor: float
    lambdas_variable: Tensor

    def __init__(self, num_classes: int, recall_range: Tuple[float, float] = (0., 1.),
                 num_anchors: int = 20, dual_rate_factor: float = 0.1,
                 surrogate_type: str = 'xent',
                 size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PrecisionRecallAUCLoss2, self).__init__(size_average, reduce, reduction)
        self.num_classes = num_classes
        self.recall_range = recall_range
        self.num_anchors = num_anchors
        self.dual_rate_factor = dual_rate_factor
        self.surrogate_type = surrogate_type

        # Create recall values.
        # Validate recall_range.
        if not 0 <= recall_range[0] <= recall_range[-1] <= 1:
            raise ValueError('recall values must obey 0 <= %f <= %f <= 1' %
                             (recall_range[0], recall_range[-1]))
        if not 0 < len(recall_range) < 3:
            raise ValueError('length of recall_range (%d) must be 1 or 2' %
                             len(recall_range))

        # Sets recall uniformly between min_recall and max_recall.
        self.recall_values = Parameter(torch.reshape(
            torch.linspace(recall_range[0], recall_range[1], num_anchors + 2)[1:-1],
            (1, 1, num_anchors)
        ), requires_grad=False)
        self.recall_delta = Parameter(self.recall_values[0, 0, 0] - recall_range[0], requires_grad=False)

        # Create lambdas.
        self.lambdas_variable = Parameter(torch.ones([1, num_classes, num_anchors]))

        # Create biases.
        self.biases = Parameter(torch.zeros([1, num_classes, num_anchors]))

        if surrogate_type == 'xent':
            self.maybe_log2 = Parameter(torch.full((1,), 2.).log_(), requires_grad=False)
        else:
            self.maybe_log2 = Parameter(torch.ones(1), requires_grad=False)


    def forward(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:

        N, C = logits.size()
        dtype = logits.dtype
        device = logits.device

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        # Maybe create weights.
        if weights is None:
            weights = torch.ones((N, C), dtype=dtype, device=device)

        # Create class priors
        class_priors = build_class_priors(targets, weights=weights)

        # Lagrange multipliers
        # Lagrange multipliers are required to be nonnegative.
        # Their gradient is reversed so that they are maximized
        # (rather than minimized) by the optimizer.
        # 2D `Tensor` of shape [num_classes, num_anchors]
        lambdas = lagrange_multiplier(self.lambdas_variable)

        # Reshape tensors
        logits = torch.unsqueeze(logits, -1)
        targets = torch.unsqueeze(targets, -1)
        weights = torch.unsqueeze(weights, -1)
        class_priors = torch.reshape(class_priors, (1, C, 1))

        # Surrogate loss.
        positive_weights = lambdas
        negative_weights = 1.
        weighted_loss = weights * weighted_surrogate_loss(
            logits=logits + self.biases,
            labels=targets,
            surrogate_type=self.surrogate_type,
            positive_weights=positive_weights,
            negative_weights=negative_weights)

        lambda_term = lambdas * class_priors * (1. - self.recall_values) * self.maybe_log2
        per_anchor_loss = weighted_loss - lambda_term
        per_class_loss = self.recall_delta * torch.sum(per_anchor_loss, 2)

        # Normalize the AUC such that a perfect score function will have AUC 1.0.
        # Because recall_range is discretized into num_anchors + 1 intervals
        # but only num_anchors terms are included in the Riemann sum, the
        # effective length of the integration interval is `delta` less than the
        # length of recall_range.
        scaled_loss = torch.div(per_class_loss,
            self.recall_range[1] - self.recall_range[0] - self.recall_delta)

        if self.reduction == 'mean':
            return scaled_loss.mean()
        elif self.reduction == 'sum':
            return scaled_loss.sum()
        return scaled_loss
