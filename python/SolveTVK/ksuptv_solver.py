"""
Solver Modified from nilearn pull request 219

"""
# Author: Eugene Belilovsky
# based on nilearn pull 219 by:
#         DOHMATOB Elvis Dopgima,
#         Gael Varoquaux,
#         Alexandre Gramfort,
#         Gaspar Pizarro,
#         Virgile Fritsch,
#         Bertrand Thirion,
#         and others.
# License: simplified BSD

import numpy as np
from .objective_functions import (spectral_norm_squared,
                                  gradient_id,
                                  logistic_loss_lipschitz_constant,
                                  squared_loss, squared_loss_grad, _unmask,
                                  logistic_loss_grad,
                                  tv_ksp_from_gradient,
                                  logistic as logistic_loss)
from .proximal_operators import (_prox_tvksp, _prox_tvksp_with_intercept)
from .fista import mfista
from Ksupport import Ksupport
KSup=Ksupport()


def tvksp_objective(X, y, w, alpha, ksp_ratio,k, mask, loss="mse",verbose=1):
    """The TV-KSP regression objective functions.

    """

    loss = loss.lower()
    if loss not in ['mse', 'logistic']:
        raise ValueError(
            "loss must be one of 'mse' or 'logistic'; got '%s'" % loss)

    if loss == "mse":
        out = squared_loss(X, y, w)
    else:
        loss_val = logistic_loss(X, y, w)
        out=loss_val
        w = w[:-1]

    grad_id = gradient_id(_unmask(w, mask), l1_ratio=ksp_ratio,square=False)
    out += alpha * tv_ksp_from_gradient(grad_id,k,square=False)

    if(verbose>0):
        tv_term = alpha*np.sum(np.sqrt(np.sum(grad_id[:-1] * grad_id[:-1],axis=0)))
        k_term = alpha*KSup.f(grad_id[-1].ravel(),k)
        r=KSup.findR(grad_id[-1].ravel(),k)[0]
        print 'Energy:',out,'loss:',loss_val,'k_sup:',k_term,' and tv_term:',tv_term,'k-r-1:',k-r-1
    return out


def tvksp_solver(X, y, alpha, ksp_ratio,k, mask, loss=None, max_iter=100,
                lipschitz_constant=None, init=None,
                prox_max_iter=5000, tol=1e-4, callback=None, verbose=1):
    """Minimizes empirical risk for TV-ksp penalized models.
"""

    # sanitize loss
    if loss not in ["mse", "logistic"]:
        raise ValueError("'%s' loss not implemented. Should be 'mse' or "
                         "'logistic" % loss)

    # shape of image box
    flat_mask = mask.ravel()
    volume_shape = mask.shape

    # We'll work on the full brain, and do the masking / unmasking
    # magic when needed
    w_size = X.shape[1] + int(loss == "logistic")

    def unmaskvec(w):
        if loss == "mse":
            return _unmask(w, mask)
        else:
            return np.append(_unmask(w[:-1], mask), w[-1])

    def maskvec(w):
        if loss == "mse":
            return w[flat_mask]
        else:
            return np.append(w[:-1][flat_mask], w[-1])

    # function to compute derivative of f1
    def f1_grad(w):
        if loss == "logistic":
            return logistic_loss_grad(X, y, w)
        else:
            return squared_loss_grad(X, y, w)

    # function to compute total energy (i.e smooth (f1) + nonsmooth (f2) parts)
    def total_energy(w):
        return tvksp_objective(X, y, w, alpha, ksp_ratio,k, mask, loss=loss,verbose=verbose)

    # Lispschitz constant of f1_grad
    if lipschitz_constant is None:
        if loss == "mse":
            lipschitz_constant = 1.05 * spectral_norm_squared(X)
        else:
            lipschitz_constant = 1.1 * logistic_loss_lipschitz_constant(X)

    # proximal operator of nonsmooth proximable part of energy (f2)
    if loss == "mse":
        def f2_prox(w, stepsize, dgap_tol, init=None):
            out, info = _prox_tvksp(
                unmaskvec(w), weight=alpha * stepsize, l1_ratio=ksp_ratio,k=k,
                dgap_tol=dgap_tol, init=unmaskvec(init),
                max_iter=prox_max_iter, verbose=verbose)
            return maskvec(out.ravel()), info
    else:
        def f2_prox(w, stepsize, dgap_tol, init=None):
            out, info = _prox_tvksp_with_intercept(
                unmaskvec(w), volume_shape, ksp_ratio, alpha * stepsize,k,
                dgap_tol, prox_max_iter, init=_unmask(
                    init[:-1], mask) if init is not None else None,
                verbose=0)
            return maskvec(out.ravel()), info

    # invoke m-FISTA solver
    w, obj, init = mfista(
        f1_grad, f2_prox, total_energy, lipschitz_constant, w_size,
        dgap_factor=(.1 + ksp_ratio) ** 2, tol=tol, init=init, verbose=verbose,
        max_iter=max_iter, callback=callback)
 
    return w, obj, init
