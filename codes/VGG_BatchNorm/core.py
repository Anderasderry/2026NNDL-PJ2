"""Curve computation utilities for loss / gradient landscape experiments."""

DEFAULT_LEARNING_RATES = [1e-3, 2e-3, 1e-4, 5e-4]


def flatten_steps(nested_list):
    return [value for epoch_values in nested_list for value in epoch_values]


def subsample_stride(stride=50):
    return slice(None, None, max(1, int(stride)))


def subsample_aligned(stride, *series):
    step = subsample_stride(stride)
    return tuple(s[step] for s in series)


def compute_min_max_curves(all_series):
    """Build max/min curves across runs keyed by learning rate."""
    flats = [flatten_steps(series) for series in all_series.values()]
    num_steps = min(len(flat) for flat in flats)
    steps = list(range(num_steps))
    max_curve, min_curve = [], []

    for step in range(num_steps):
        values_at_step = [flat[step] for flat in flats]
        max_curve.append(max(values_at_step))
        min_curve.append(min(values_at_step))

    return steps, min_curve, max_curve


def compute_grad_predictiveness(flat_grads):
    if len(flat_grads) < 2:
        return [], []
    deltas = [abs(flat_grads[i] - flat_grads[i - 1]) for i in range(1, len(flat_grads))]
    return list(range(1, len(flat_grads))), deltas


def compute_max_diff_curve(min_curve, max_curve):
    return [max_val - min_val for max_val, min_val in zip(max_curve, min_curve)]
