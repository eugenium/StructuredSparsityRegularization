"""Implementation of multiple proximal operators for TV-L1, S-LASSO, etc.

"""
# Author: DOHMATOB Elvis Dopgima,
#         VAROQUAUX Gael,
#         GRAMFORT Alexandre,
# License: simplified BSD

from math import sqrt
import numpy as np
from .objective_functions import tv_ksp_from_gradient, div_id, gradient_id
import KsupportTV as KTV
KSup=KTV.Ksupport()


def _projector_on_tvksp_dual(grad, l1_ratio,k):
    """
    modifies IN PLACE the gradient + id to project it
    on the l21 unit ball in the gradient direction and the L1 ball in the
    identity direction
    """

    # The l21 ball for the gradient direction
    if l1_ratio < 1.:
        end = len(grad) - int(l1_ratio > 0.)
        norm = np.sqrt(np.sum(grad[:end] * grad[:end], 0))
        norm.clip(1., out=norm)  # set everythx < 1 to 1
        for grad_comp in grad[:end]:
            grad_comp /= norm

    # The L1 ball for the identity direction
    if l1_ratio > 0.:
       # grad[-1] = KSup.ProxBall(grad[-1].ravel(),1.0,k).reshape(grad[-1].shape) #projection onto the dual is the prox of the primal
        grad[-1] = KSup.ProxBall(grad[-1].ravel(),1.0,k,Dual=True).reshape(grad[-1].shape)

    return grad


def _dual_gap_prox_tvksp(input_img_norm, new, gap, weight, l1_ratio=1.,k=10):
    """
    Dual gap of total variation denoising
    see "Total variation regularization for fMRI-based prediction of behavior",
    by Michel et al. (2011) for a derivation of the dual gap
    """
    tv_new = tv_ksp_from_gradient(gradient_id(new, l1_ratio=l1_ratio,square=False),k) #CHANGED
    gap = gap.ravel()
    d_gap = np.dot(gap, gap) + 2 * weight * tv_new - input_img_norm + (
        new * new).sum()
    return 0.5 * d_gap


def _objective_function_prox_tvksp(input_img, output_img, gradient, weight,k):
    diff = (input_img - output_img).ravel()
    return (.5 * (diff * diff).sum() + weight * tv_ksp_from_gradient(gradient,k))


def _prox_tvksp(input_img, l1_ratio=.05,k=10, weight=50, dgap_tol=5.e-5, x_tol=None,
              max_iter=200, check_gap_frequency=4, val_min=None, val_max=None,
              verbose=False, fista=True, init=None):
    """
    Compute the TV-L1 proximal (ie total-variation +l1 denoising) on 3d images.

    Find the argmin `res` of
        1/2 * ||im - res||^2 + weight * TVl1(res),

    Parameters
    ----------
    input_img : ndarray of floats (2-d or 3-d)
        Input data to be denoised. `im` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.

    weight : float, optional
        Denoising weight. The greater ``weight``, the more denoising (at
        the expense of fidelity to ``input``)

    dgap_tol : float, optional
        Precision required. The distance to the exact solution is computed
        by the dual gap of the optimization problem and rescaled by the
        squared l2 norm of the image (for contrast invariance).

    x_tol : float or None, optional
        The maximal relative difference between input and output. If
        specified, this specifies a stopping criterion on x, rather than
        the dual gap.

    max_iter : int, optional
        Maximal number of iterations used for the optimization.

    val_min : None or float, optional
        An optional lower bound constraint on the reconstructed image.

    val_max : None or float, optional
        An optional upper bound constraint on the reconstructed image.

    verbose : bool, optional
        If True, print the dual gap of the optimization

    fista : bool, optional
        If True, uses a FISTA loop to perform the optimization.
        if False, uses an ISTA loop.

    callback : callable
        Callable that takes the local variables at each
        steps. Useful for tracking.

    init : array of shape shape as im
        Starting point for the optimization.

    Returns
    -------
    out : ndarray
        TV-l1-denoised image.

    Notes
    -----
    The principle of total variation denoising is explained in
    http://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This function implements the FISTA (Fast Iterative Shrinkage
    Thresholding Algorithm) algorithm of Beck et Teboulle, adapted to
    total variation denoising in "Fast gradient-based algorithms for
    constrained total variation image denoising and deblurring problems"
    (2009).

    For details on implementing the bound constraints, read the aforementioned
    Beck and Teboulle paper.
    """
    weight = float(weight)
    input_img_flat = input_img.view()
    input_img_flat.shape = input_img.size
    input_img_norm = np.dot(input_img_flat, input_img_flat)
    if not input_img.dtype.kind == 'f':
        input_img = input_img.astype(np.float)
    shape = [len(input_img.shape) + 1] + list(input_img.shape)
    grad_im = np.zeros(shape)
    grad_aux = np.zeros(shape)
    t = 1.
    i = 0
    lipschitz_constant = 1.1 * (4 * input_img.ndim * (1 - l1_ratio)
                                ** 2 + l1_ratio ** 2)

    # negated_output is the negated primal variable in the optimization
    # loop
    if init is None:
        negated_output = -input_img
    else:
        negated_output = -init

    # Clipping values for the inner loop
    negated_val_min = np.inf
    negated_val_max = -np.inf
    if val_min is not None:
        negated_val_min = -val_min
    if val_max is not None:
        negated_val_max = -val_max
    if True or (val_min is not None or val_max is not None):
        # With bound constraints, the stopping criterion is on the
        # evolution of the output
        negated_output_old = negated_output.copy()
    grad_tmp = None
    old_dgap = np.inf
    dgap = np.inf

    # A boolean to control if we are going to do a fista step
    fista_step = fista

    while i < max_iter:
        grad_tmp = gradient_id(negated_output, l1_ratio=l1_ratio) #CHANGED
        grad_tmp *= 1. / (lipschitz_constant * weight)
        grad_aux += grad_tmp
        grad_tmp = _projector_on_tvksp_dual(  #CHANGED
            grad_aux, l1_ratio,k
            )

        # Careful, in the next few lines, grad_tmp and grad_aux are a
        # view on the same array, as _projector_on_tvl1_dual returns a view
        # on the input array
        t_new = 0.5 * (1. + sqrt(1. + 4. * t * t))
        t_factor = (t - 1.) / t_new
        if fista_step:
            grad_aux = (1 + t_factor) * grad_tmp - t_factor * grad_im
        else:
            grad_aux = grad_tmp
        grad_im = grad_tmp
        t = t_new
        gap = weight * div_id(grad_aux, l1_ratio=l1_ratio) #CHANGED

        # Compute the primal variable
        negated_output = gap - input_img
        #negated_output=-KSup.ProxBall(-negated_output,1.0,k,Dual=True).reshape(negated_output.shape)
        if (val_min is not None or val_max is not None):
            negated_output = negated_output.clip(negated_val_max,
                                                 negated_val_min,
                                                 out=negated_output)
        if (i % check_gap_frequency) == 0:
            if x_tol is None:
                # Stopping criterion based on the dual gap
                if val_min is not None or val_max is not None:
                    # We need to recompute the dual variable
                    gap = negated_output + input_img
                old_dgap = dgap
                dgap = _dual_gap_prox_tvksp(input_img_norm, -negated_output, #CHANGED
                                           gap, weight, l1_ratio=l1_ratio,k=k)
                if verbose:
                    print '\tProxTVl1: Iteration % 2i, dual gap: % 6.3e' % (
                        i, dgap)
                if dgap < dgap_tol:
                    break
                if old_dgap < dgap:
                    # M-FISTA strategy: switch to an ISTA to have
                    # monotone convergence
                    fista_step = False
                elif fista:
                    fista_step = True
            else:
                # Stopping criterion based on x_tol
                diff = np.max(np.abs(negated_output_old - negated_output))
                diff /= np.max(np.abs(negated_output))
                if verbose:
                    print ('\tProxTVl1 iteration % 2i, relative difference:'
                           ' % 6.3e, energy: % 6.3e') % (
                        i, diff, _objective_function_prox_tvksp( #CHANGED
                                   input_img, -negated_output, gradient_id(
                                       negated_output, l1_ratio=l1_ratio),
                            weight,k))
                if diff < x_tol:
                    break
                negated_output_old = negated_output
        i += 1

    # Compute the primal variable, however, here we must use the ista
    # value, not the fista one
    output = input_img - weight * div_id(grad_im, l1_ratio=l1_ratio) #CHANGED
    if (val_min is not None or val_max is not None):
        output = output.clip(val_min, val_max, out=output)
    return output, dict(converged=(i < max_iter))


def _prox_tvksp_with_intercept(w, shape, l1_ratio, weight,k, dgap_tol,
                              max_iter=5000, init=None, verbose=False):
    """
    Computation of TV-L1 prox, taking into account the intercept.

    Parameters
    ----------
    weight : float
       Weight in prox. This would be something like `alpha_ * stepsize`,
       where `alpha_` is the effective (i.e. re-scaled) alpha.

    w : ndarray, shape (w_size,)
        The point at which the prox is being computed

    init : ndarray, shape (w_size - 1,), optional (default None)
        Initialization vector for the prox.

    dgap_tol : float
        Dual-gap tolerance for TV-L1 prox operator approximation loop.

    """

    init = init.reshape(shape) if not init is None else init
    out, prox_info = _prox_tvksp(
        w[:-1].reshape(shape), weight=weight,
        l1_ratio=l1_ratio,k=k, dgap_tol=dgap_tol, init=init, max_iter=max_iter,
        verbose=verbose)#,x_tol=dgap_tol)

    return np.append(out, w[-1]), prox_info
