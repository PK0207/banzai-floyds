import numpy as np
from scipy.optimize import minimize

# This whole framework is adapted from Zackay et al. 2017


def matched_filter_signal(data, error, weights):
    return (data * weights / error / error).sum()


def matched_filter_normalization(error, weights):
    return ((weights * weights / error / error).sum()) ** 0.5


def matched_filter_metric(theta, data, error, weight_function, weights_jacobian_function,
                          weights_hessian_function, x, *args):
    weights = weight_function(theta, x, *args)
    metric = matched_filter_signal(data, error, weights)
    metric /= matched_filter_normalization(error, weights)
    return metric


def matched_filter_signal_jacobian(theta, x, data, error, weights_jacobian_function, *args):
    return np.array([(weights_jacobian_function(theta, x, i, *args) * data / error / error).sum()
                     for i in range(len(theta))])


def matched_filter_normalization_jacobian(theta, x, weights, error, weights_jacobian_function, normalization, *args):
    # n' = sum weights * weights'/ sigma^2 / n
    return np.array([(weights * weights_jacobian_function(theta, x, i, *args) / error / error).sum() / normalization
                     for i in range(len(theta))])


def matched_filter_jacobian(theta, data, error, weights_function, weights_jacobian_function,
                            weights_hessian_function, x, *args):
    # First derivative of the matched filter metric
    # metric is s / n
    # jacobian is (n s' - s n') / n^2
    weights = weights_function(theta, x, *args)
    normalization = matched_filter_normalization(error, weights)
    signal = matched_filter_signal(data, error, weights)

    # s' = sum Weights' data / error^2
    signal_jacobian = matched_filter_signal_jacobian(theta, x, data, error, weights_jacobian_function, *args)

    normalization_jacobian = matched_filter_normalization_jacobian(theta, x, weights, error, weights_jacobian_function,
                                                                   normalization, *args)
    return (normalization * signal_jacobian - signal * normalization_jacobian) / normalization / normalization


def matched_filter_hessian(theta, data, error, weights_function, weights_jacobian_function, weights_hessian_function, x,
                           *args):
    # The Hessian is n⁻⁴ (n² (∂ⱼn ∂ᵢs + n ∂ⱼ∂ᵢs -s ∂ⱼ∂ᵢn - ∂ᵢn ∂ⱼs) - 2 n ∂ⱼn (n ∂ᵢs - s ∂ᵢn))
    weights = weights_function(theta, x, *args)
    signal = matched_filter_signal(data, error, weights)
    normalization = matched_filter_normalization(error, weights)

    signal_jacobian = matched_filter_signal_jacobian(theta, x, data, error, weights_jacobian_function, *args)
    normalization_jacobian = matched_filter_normalization_jacobian(theta, x, weights, error,
                                                                   weights_jacobian_function, normalization,
                                                                   *args)

    weights_hessian = np.zeros((theta.size, theta.size, *data.shape))
    for i in range(len(theta)):
        for j in range(len(theta)):
            if weights_hessian[j, i].sum() != 0.0:
                weights_hessian[i, j] = weights_hessian[j, i]
            else:
                weights_hessian[i, j] = weights_hessian_function(theta, x, i, j, *args)

    signal_hessian = np.zeros((theta.size, theta.size))
    for i in range(len(theta)):
        for j in range(len(theta)):
            if signal_hessian[j, i] != 0.0:
                signal_hessian[i, j] = signal_hessian[j, i]
            else:
                signal_hessian[i, j] = (weights_hessian[i, j] * data / error / error).sum()

    normalization_hessian = np.zeros((theta.size, theta.size))
    for i in range(len(theta)):
        for j in range(len(theta)):
            if normalization_hessian[j, i] != 0.0:
                normalization_hessian[i, j] = normalization_hessian[j, i]
            else:
                # the hessian of the normalization
                # ∂ⱼn = Σ(weights * ∂ⱼweights / σ²)/n
                # ∂ᵢ∂ⱼn = n⁻² (Σ((∂ᵢweights * ∂ⱼweights  + weights * ∂ᵢ∂ⱼweights)/ σ²) * n - ∂ᵢn ∂ⱼn)
                first_term = weights_jacobian_function(theta, x, i, *args)
                first_term *= weights_jacobian_function(theta, x, j, *args)
                first_term += weights * weights_hessian[i, j]
                first_term /= error * error
                normalization_hessian[i, j] = first_term.sum() * normalization
                normalization_hessian[i, j] -= normalization_jacobian[i] * normalization_jacobian[j]
                normalization_hessian[i, j] /= normalization * normalization

    filter_hessian = np.zeros((theta.size, theta.size))
    for i in range(len(theta)):
        for j in range(len(theta)):
            # Short circuit because hessian has to be symmetric
            if filter_hessian[j, i] != 0.0:
                filter_hessian[i, j] = filter_hessian[j, i]
                continue
            # Start with this: (∂ⱼn ∂ᵢs + n ∂ⱼ∂ᵢs -s ∂ⱼ∂ᵢn - ∂ᵢn ∂ⱼs)
            filter_hessian[i, j] = normalization_jacobian[j] * signal_jacobian[i] + normalization * signal_hessian[j, i]
            filter_hessian -= signal * normalization_hessian[j, i] - normalization_jacobian[i] * signal_jacobian[j]
            filter_hessian[i, j] *= normalization ** 2.0
            # 2 n ∂ⱼn (n ∂ᵢs - s ∂ᵢn)
            term2 = normalization * signal_jacobian[i] - signal * normalization_jacobian[i]
            filter_hessian[i, j] -= 2.0 * normalization * normalization_jacobian[j] * term2
            filter_hessian[i, j] *= normalization ** -4.0

    return filter_hessian


def maximize_match_filter(initial_guess, data, error, weight_function, x, weights_jacobian_function=None,
                          weights_hessian_function=None,  args=None):
    if weights_hessian_function is None and weights_jacobian_function is None:
        best_fit = minimize(lambda *params: -matched_filter_metric(*params), initial_guess,
                            args=(data, error, weight_function, weights_jacobian_function, weights_hessian_function, x,
                                  *args), method='Powell')
    elif weights_hessian_function is None:
        best_fit = minimize(lambda *params: -matched_filter_metric(*params), initial_guess,
                            args=(data, error, weight_function, weights_jacobian_function, weights_hessian_function, x,
                                  *args),
                            method='BFGS', jac=lambda *params: -matched_filter_jacobian(*params))
    else:
        best_fit = minimize(lambda *params: -matched_filter_metric(*params), initial_guess,
                            args=(data, error, weight_function, weights_jacobian_function, weights_hessian_function,
                                  x, *args),
                            method='Newton-CG', hess=lambda *params: -matched_filter_hessian(*params),
                            jac=lambda *params: -matched_filter_jacobian(*params))
    return best_fit.x
