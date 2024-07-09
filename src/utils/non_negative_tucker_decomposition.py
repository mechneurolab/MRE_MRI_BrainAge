#%%
import tensorly as tl
import numpy as np
import warnings
from scipy.optimize import brentq
from collections.abc import Iterable
from tensorly.tenalg import svd_interface, multi_mode_dot
from tensorly import unfold
from tensorly.tenalg.proximal import hals_nnls
from tensorly.tucker_tensor import tucker_to_tensor, TuckerTensor
# %%
def tucker_normalize(tucker_tensor):
    """Returns tucker_tensor with factors normalised to unit length with the normalizing constants absorbed into
    `core`.
    Parameters
    ----------
    tucker_tensor : tl.TuckerTensor or (core, factors)
        core tensor and list of factor matrices
    Returns
    -------
    TuckerTensor((core, factors))
    """
    core, factors = tucker_tensor
    normalized_factors = []
    for i, factor in enumerate(factors):
        scales = tl.norm(factor, axis=0)
        scales_non_zero = tl.where(
            scales == 0, tl.ones(tl.shape(scales), **tl.context(factor)), scales
        )
        core = core * tl.reshape(
            scales, (1,) * i + (-1,) + (1,) * (tl.ndim(core) - i - 1)
        )
        normalized_factors.append(factor / tl.reshape(scales_non_zero, (1, -1)))
    return TuckerTensor((core, normalized_factors))
# %%
def fista(
    UtM,
    UtU,
    x=None,
    n_iter_max=100,
    non_negative=True,
    sparsity_coef=0,
    lr=None,
    tol=10e-8,
):
    """
    Fast Iterative Shrinkage Thresholding Algorithm (FISTA)
    Computes an approximate (nonnegative) solution for Ux=M linear system.
    Parameters
    ----------
    UtM : ndarray
        Pre-computed product of the transposed of U and M
    UtU : ndarray
        Pre-computed product of the transposed of U and U
    x : init
       Default: None
    n_iter_max : int
        Maximum number of iteration
        Default: 100
    non_negative : bool, default is False
                   if True, result will be non-negative
    lr : float
        learning rate
        Default : None
    sparsity_coef : float or None
    tol : float
        stopping criterion
    Returns
    -------
    x : approximate solution such that Ux = M
    Notes
    -----
    We solve the following problem :math: `1/2 ||m - Ux ||_2^2 + \\lambda |x|_1`
    Reference
    ----------
    [1] : Beck, A., & Teboulle, M. (2009). A fast iterative
          shrinkage-thresholding algorithm for linear inverse problems.
          SIAM journal on imaging sciences, 2(1), 183-202.
    """
    if sparsity_coef is None:
        sparsity_coef = 0

    if x is None:
        x = tl.zeros(tl.shape(UtM), **tl.context(UtM))
    if lr is None:
        lr = 1 / (tl.truncated_svd(UtU)[1][0])
    # Parameters
    momentum_old = tl.tensor(1.0)
    norm_0 = 0.0
    x_update = tl.copy(x)
    # print(tl.shape(x_update))
    for iteration in range(n_iter_max):
        constraint = np.stack([(~np.eye(tl.shape(UtM)[0], dtype=bool)) for i in range(tl.shape(UtM)[1])], axis=1).astype(float)
        if isinstance(UtU, list):
            x_gradient = (
                -UtM
                + tl.tenalg.multi_mode_dot(x_update, UtU, transpose=False)
                + sparsity_coef*constraint
            )

        else:
            x_gradient = -UtM + tl.dot(UtU, x_update) + sparsity_coef*constraint

        if non_negative is True:
            x_gradient = tl.where(lr * x_gradient < x_update, x_gradient, x_update / lr)

        x_new = x_update - lr * x_gradient
        momentum = (1 + tl.sqrt(1 + 4 * momentum_old**2)) / 2
        x_update = x_new + ((momentum_old - 1) / momentum) * (x_new - x)
        momentum_old = momentum
        x = tl.copy(x_new)
        norm = tl.norm(lr * x_gradient)
        if iteration == 1:
            norm_0 = norm
        if norm < tol * norm_0:
            break
    return x
# %%
def initialize_tucker(
    tensor,
    rank,
    modes,
    random_state,
    init="svd",
    svd="truncated_svd",
    non_negative=False,
    mask=None,
    svd_mask_repeats=5,
):
    """
    Initialize core and factors used in `tucker`.
    The type of initialization is set using `init`. If `init == 'random'` then
    initialize factor matrices using `random_state`. If `init == 'svd'` then
    initialize the `m`th factor matrix using the `rank` left singular vectors
    of the `m`th unfolding of the input tensor.
    Parameters
    ----------
    tensor : ndarray
    rank : int
           number of components
    modes : int list
    random_state : {None, int, np.random.RandomState}
    init : {'svd', 'random', cptensor}, optional
    svd : str, default is 'truncated_svd'
          function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    non_negative : bool, default is False
        if True, non-negative factors are returned
    Returns
    -------
    core    : ndarray
              initialized core tensor
    factors : list of factors
    """
    # Initialisation
    if init == "svd":
        factors = []
        for index, mode in enumerate(modes):
            mask_unfold = None if mask is None else unfold(mask, mode)
            U, _, _ = svd_interface(
                unfold(tensor, mode),
                n_eigenvecs=rank[index],
                method=svd,
                non_negative=non_negative,
                mask=mask_unfold,
                n_iter_mask_imputation=svd_mask_repeats,
                random_state=random_state,
            )

            factors.append(U)
        # The initial core approximation is needed here for the masking step
        core = multi_mode_dot(tensor, factors, modes=modes, transpose=True)

    elif init == "random":
        rng = tl.check_random_state(random_state)
        core = tl.tensor(
            rng.random_sample(rank) + 0.01, **tl.context(tensor)
        )  # Check this
        factors = [
            tl.tensor(rng.random_sample(s), **tl.context(tensor))
            for s in zip(tl.shape(tensor), rank)
        ]

    else:
        (core, factors) = init

    if non_negative is True:
        factors = [tl.abs(f) for f in factors]
        core = tl.abs(core)

    return core, factors
# %%
def validate_tucker_rank(tensor_shape, rank="same", rounding="round", fixed_modes=None):
    r"""Returns the rank of a Tucker Decomposition
    Parameters
    ----------
    tensor_shape : tupe
        shape of the tensor to decompose
    rank : {'same', float, tuple, int}, default is same
        way to determine the rank, by default 'same'
        if 'same': rank is computed to keep the number of parameters (at most) the same
        if float, computes a rank so as to keep rank percent of the original number of parameters
        if int or tuple, just returns rank
    rounding = {'round', 'floor', 'ceil'}
    fixed_modes : int list or None, default is None
        if not None, a list of modes for which the rank will be the same as the original shape
        e.g. if i in fixed_modes, then rank[i] = tensor_shape[i]
    Returns
    -------
    rank : int tuple
        rank of the decomposition
    Notes
    -----
    For a fractional input rank, I want to find a Tucker rank such that:
    n_param_decomposition = rank*n_param_tensor
    In particular, for an input of size I_1, ..., I_N:
    I find a value c such that the rank will be (c I_1, ..., c I_N)
    We have sn_param_tensor = I_1 x ... x I_N
    We look for a Tucker decomposition of rank (c I_1, ..., c I_N )
    This decomposition will have the following n_params:
    For the core : \prod_k c I_k = c^N \prod I_k = c^N n_param_tensor
    For the factors : \sum_k c I_k^2
    In other words we want to solve:
    c^N n_param_tensor + \sum_k c I_k^2 = rank*n_param_tensor
    """
    if rounding == "ceil":
        rounding_fun = np.ceil
    elif rounding == "floor":
        rounding_fun = np.floor
    elif rounding == "round":
        rounding_fun = np.round
    else:
        raise ValueError(f"Rounding should be round, floor or ceil, but got {rounding}")

    # rank is 'same' or float: choose rank so as to preserve a fraction of the original #parameters
    if rank == "same":
        rank = float(1)

    if isinstance(rank, float):
        n_modes_compressed = len(tensor_shape)
        n_param_tensor = np.prod(tensor_shape)

        if fixed_modes is not None:
            tensor_shape = list(tensor_shape)

            # sorted to be careful with the order when popping and reinserting to not remove/add at wrong index.
            # list (mode, shape) that we removed as they will be kept the same, rank[i] =
            fixed_modes = [
                (mode, tensor_shape.pop(mode))
                for mode in sorted(fixed_modes, reverse=True)
            ][::-1]

            # number of parameters coming from the fixed modes (these don't have a variable size as a fun of fraction_param)
            n_fixed_params = np.sum(
                [s**2 for _, s in fixed_modes]
            )  # size of the factors
            n_modes_compressed -= len(fixed_modes)
        else:
            n_fixed_params = 0

        # Doesn't contain fixed_modes, those factors are accounted for in fixed_params
        squared_dims = np.sum([s**2 for s in tensor_shape])

        fun = (
            lambda x: n_param_tensor * x**n_modes_compressed
            + squared_dims * x
            + n_fixed_params * x
            - rank * n_param_tensor
        )
        fraction_param = brentq(fun, 0.0, max(rank, 1.0))
        rank = [max(int(rounding_fun(s * fraction_param)), 1) for s in tensor_shape]

        if fixed_modes is not None:
            for mode, size in fixed_modes:
                rank.insert(mode, size)

    elif isinstance(rank, int):
        n_modes = len(tensor_shape)
        message = f"Given only one int for 'rank' for decomposition a tensor of order {n_modes}. Using this rank for all modes."
        warnings.warn(message, RuntimeWarning)
        if fixed_modes is None:
            rank = [rank] * n_modes
        else:
            rank = [
                rank if i not in fixed_modes else s
                for (i, s) in enumerate(tensor_shape)
            ]  # *n_mode

    return rank
# %%
def non_negative_tucker_hals(
                            tensor,
                            rank,
                            n_iter_max=100,
                            init="svd",
                            svd="truncated_svd",
                            tol=1e-8,
                            sparsity_coefficients=None,
                            core_sparsity_coefficient=None,
                            fixed_modes=None,
                            random_state=None,
                            verbose=False,
                            normalize_factors=False,
                            return_errors=False,
                            exact=False,
                            algorithm="fista"):

    rank = validate_tucker_rank(tl.shape(tensor), rank=rank)
    n_modes = tl.ndim(tensor)
    if sparsity_coefficients is None or not isinstance(sparsity_coefficients, Iterable):
        sparsity_coefficients = [sparsity_coefficients] * n_modes

    if fixed_modes is None:
        fixed_modes = []

    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn(
            "You asked for fixing the last mode, which is not supported. The last mode will not be fixed."
            " Consider using tl.moveaxis() to permute it to another position and keep it fixed there."
        )
        fixed_modes.remove(tl.ndim(tensor) - 1)

    # Avoiding errors
    for fixed_value in fixed_modes:
        sparsity_coefficients[fixed_value] = None
    # Generating the mode update sequence
    modes = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    nn_core, nn_factors = initialize_tucker(
        tensor,
        rank,
        modes,
        init=init,
        svd=svd,
        random_state=random_state,
        non_negative=True,
    )


    # initialisation - declare local variables
    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []

    # Iterate over one step of NTD
    for iteration in range(n_iter_max):
        # One pass of least squares on each updated mode
        for mode in modes:
            # Computing Hadamard of cross-products
            pseudo_inverse = nn_factors.copy()
            for i, factor in enumerate(nn_factors):
                if i != mode:
                    pseudo_inverse[i] = tl.dot(tl.conj(tl.transpose(factor)), factor)
            # UtU
            core_cross = multi_mode_dot(nn_core, pseudo_inverse, skip=mode)
            UtU = tl.dot(unfold(core_cross, mode), tl.transpose(unfold(nn_core, mode)))

            # UtM
            tensor_cross = multi_mode_dot(tensor, nn_factors, skip=mode, transpose=True)
            MtU = tl.dot(
                unfold(tensor_cross, mode), tl.transpose(unfold(nn_core, mode))
            )
            UtM = tl.transpose(MtU)

            # Call the hals resolution with nnls, optimizing the current mode
            nn_factor, _, _, _ = hals_nnls(
                UtM,
                UtU,
                tl.transpose(nn_factors[mode]),
                n_iter_max=100,
                sparsity_coefficient=sparsity_coefficients[mode],
                exact=exact,
            )
            nn_factors[mode] = tl.transpose(nn_factor)
        # updating core
        if algorithm == "fista":
            pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
            core_estimation = multi_mode_dot(tensor, nn_factors, transpose=True)
            learning_rate = 1

            for MtM in pseudo_inverse:
                learning_rate *= 1 / (tl.truncated_svd(MtM)[1][0])
            nn_core = fista(
                core_estimation,
                pseudo_inverse,
                x=nn_core,
                n_iter_max=n_iter_max,
                sparsity_coef=core_sparsity_coefficient,
                lr=learning_rate,
            )
    #   if algorithm == "active_set":
    #       pseudo_inverse[-1] = tl.dot(tl.transpose(nn_factors[-1]), nn_factors[-1])
    #       core_estimation_vec = tl.base.tensor_to_vec(
    #           tl.tenalg.mode_dot(
    #               tensor_cross, tl.transpose(nn_factors[modes[-1]]), modes[-1]
    #           )
    #       )
    #       pseudo_inverse_kr = tl.tenalg.kronecker(pseudo_inverse)
    #       vectorcore = active_set_nnls(
    #           core_estimation_vec, pseudo_inverse_kr, x=nn_core, n_iter_max=n_iter_max
    #       )
    #       nn_core = tl.reshape(vectorcore, tl.shape(nn_core))

        # Adding the l1 norm value to the reconstruction error
        sparsity_error = 0
        for index, sparse in enumerate(sparsity_coefficients):
            if sparse:
                sparsity_error += 2 * (sparse * tl.norm(nn_factors[index], order=1))
        # error computation
        rec_error = (
            tl.norm(tensor - tucker_to_tensor((nn_core, nn_factors)), 2) / norm_tensor
        )
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print(
                    f"reconstruction error={rec_errors[-1]}, variation={rec_errors[-2] - rec_errors[-1]}."
                )

            if tol and abs(rec_errors[-2] - rec_errors[-1]) < tol:
                if verbose:
                    print(f"converged in {iteration} iterations.")
                break
        if normalize_factors:
            nn_core, nn_factors = tucker_normalize((nn_core, nn_factors))
    tensor = TuckerTensor((nn_core, nn_factors))
    if return_errors:
        return tensor, rec_errors
    else:
        return tensor
# %%
# tensor generation
# array = np.random.randint(1000, size=(10, 30, 40))
# tensor = tl.tensor(array, dtype='float')
# rank = [3,5,5]
# tensor_decomp = non_negative_tucker_hals(tensor, rank, core_sparsity_coefficient=0.05, normalize_factors=False)
# # %%
