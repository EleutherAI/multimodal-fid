import numpy as np


def symmetric_matrix_square_root(x, eps=1e-10):
    """Square root of matrix x such that x = a @ a
    x _must_ be both symmetric and positive semi-definite or else this will be nonsense
    """
    u, s, v = np.linalg.svd(x)
    si = np.where((np.abs(s) <= eps), s, np.sqrt(s))
    return u @ np.diag(si) @ v


def trace_sqrt_product(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.
    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
      => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
      => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
      => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                    = sum(sqrt(eigenvalues(A B B A)))
                                    = sum(eigenvalues(sqrt(A B B A)))
                                    = trace(sqrt(A B B A))
                                    = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.
    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma
    Returns:
      The trace of the positive square root of sigma*sigma_v
    """
    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = symmetric_matrix_square_root(sigma)
    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = sqrt_sigma @ (sigma_v @ sqrt_sigma)
    return np.trace(symmetric_matrix_square_root(sqrt_a_sigmav_a))


def multimodal_frechet_distance(y, y_hat, x):
    '''
    CFID metric implementation, according to the formula described in the paper;
    https://arxiv.org/abs/2103.11521
    The formula:
    Given (x,y)~N(m_xy,C) and (x,y_h)~N(m_xy_h,C_h)
    Assume their joint Gaussian distribution:
    C = [[C_xx,   C_xy]
         [C_yx,   C_yy]]
    C_h = [[C_xx,   C_xy_h]
         [C_y_hx, C_y_hy_h]]
    m_xy = mean(x,y)
    m_xy_h = mean(x,y_h)
    Denote:
    C_y|x   = C_yy - C_yx @ (C_xx^-1) @ C_xy
    C_y_h|x = C_y_hy_h - C_y_hx @ (C_xx^-1) @ C_xy_h
    m_y     = mean(y)
    m_y_h   = mean(y_h)
    CFID((x,y), (x,y_h)) = ||m_y - m_y_h||^2 + Tr((C_yx-C_y_hx) @ (C_xx^-1) @ (C_xy-C_x_y_h)) + \
                                             + Tr(C_y|x + C_y_h|x) -2*Tr((C_y|x @ (C_y_h|x^(1/2)) @ C_y|x)^(1/2))
    The arguments:
    y_true    = [N,k1]
    y_predict = [N,k2]
    x_true    = [N,k3]
    embedding - Functon that transform [N,ki] -> [N,m], 'no_embedding' might be consider to used, if you working with same dimensions activations.
    estimator - Covariance estimator. Default is sample covariance estimator.
                The estimator might be switched to other estimators. Remmember that other estimator must support 'invert' argument
    '''
    # assert ((y_predict.shape[0] == y_true.shape[0]) and (y_predict.shape[0] == x_true.shape[0]))
    # assert ((y_predict.shape[1] == y_true.shape[1]) and (y_predict.shape[1] == x_true.shape[1]))
    # Sample means
    # y_dim = y.shape[1]
    x_dim = x.shape[1]
    y_mean = np.mean(y, axis=0)
    y_hat_mean = np.mean(y_hat, axis=0)
    # x_mean = np.mean(x, axis=0)
    # sample covariance
    # C_ŷx. size is (y_hat_dim + x_dim, y_hat_dim + x_dim)
    y_hat_x_cov_full = np.cov(y_hat, x, rowvar=False)
    y_hat_x_cov = y_hat_x_cov_full[:-x_dim, -x_dim:] # size is (y_hat_dim, x_dim)
    # C_xŷ
    x_y_hat_cov = y_hat_x_cov.T
    # C_yx, size is (y_dim + x_dim, y_dim + x_dim)
    y_x_cov_full = np.cov(y, x, rowvar=False) 
    y_x_cov = y_x_cov_full[:-x_dim, -x_dim:] # size is (x_dim, y_dim)
    # C_xy
    x_y_cov = y_x_cov.T 
    # C_ŷŷ
    y_hat_cov = np.cov(y_hat, rowvar=False)
    # C_yy
    y_cov = np.cov(y, rowvar=False)
    # (C_xx)^(-1)
    x_cov_inverse = np.linalg.pinv(np.cov(x, rowvar=False))
    # C_(yy|x)
    y_cond_cov = y_cov - (y_x_cov @ (x_cov_inverse @ x_y_cov))
    # C_(ŷŷ|x)
    y_hat_cond_cov = y_hat_cov - (y_hat_x_cov @ (x_cov_inverse @ x_y_hat_cov))
    # Distance between Gaussians
    m_dist = np.sum(np.square(y_mean - y_hat_mean))
    cov_dist1 = np.trace(
        (y_x_cov - y_hat_x_cov )@ x_cov_inverse @ (x_y_cov - x_y_hat_cov)
    )
    cov_dist2 = np.trace(y_cond_cov + y_hat_cond_cov) - 2 * trace_sqrt_product(y_cond_cov, y_hat_cond_cov)
    return m_dist + cov_dist1 + cov_dist2
