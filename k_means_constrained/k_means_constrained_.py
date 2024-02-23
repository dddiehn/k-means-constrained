"""k-means-constrained"""

# Authors: Josh Levy-Kramer <josh@levykramer.co.uk>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
# License: BSD 3 clause

import warnings
import numpy as np
import scipy.sparse as sp
from .sklearn_import.metrics.pairwise import euclidean_distances
from .sklearn_import.utils.extmath import row_norms, squared_norm, cartesian
from .sklearn_import.utils.validation import check_array, check_random_state, as_float_array, check_is_fitted
from joblib import Parallel
from joblib import delayed

# Internal scikit learn methods imported into this project
from k_means_constrained.sklearn_import.cluster._k_means import _centers_dense, _centers_dense_two_data_sets, _centers_sparse
from k_means_constrained.sklearn_import.cluster.k_means_ import _validate_center_shape, _tolerance, KMeans, \
    _init_centroids_two_data_sets, _init_centroids

from ortools.graph.python.min_cost_flow import SimpleMinCostFlow


def k_means_constrained(X,
                        n_clusters,
                        size_min        = None,
                        size_max        = None,
                        init            = 'k-means++',
                        n_init          = 10,
                        max_iter        = 300,
                        verbose         = False,
                        tol             = 1e-4,
                        random_state    = None,
                        copy_x          = True,
                        n_jobs          = 1,
                        return_n_iter   = False,
                        distance_metric = euclidean_distances,
                        **kwargs):
    """K-Means clustering with minimum and maximum cluster size constraints.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The observations to cluster.

    size_min : int, optional, default: None
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied

    size_max : int, optional, default: None
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : boolean, optional
        Verbosity mode.

    tol : float, optional
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy_x : boolean, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    distance_metric : callable, default = euclidean_distances
        Distance metric used for calculating distance between points in the
        dataset. Defaults to the euclidean_distances function from
        sklearn.metrics.pairwise.
        If defining a custom function, it must accept two arguments: X, and
        cluster centers, in that order.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.

    """
    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    # Validate init array
    if hasattr(init, '__array__'):
        init = check_array(init, dtype=X.dtype.type, copy=True)
        _validate_center_shape(X, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    if not sp.issparse(X):
        X_mean = X.mean(axis=0)
        # The copy was already done above
        X -= X_mean

        if hasattr(init, '__array__'):
            init -= X_mean

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None

    if n_jobs == 1:
        # For a single thread, less memory is needed if we just store one set
        # of the best results (as opposed to one set per run per thread).
        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_constrained_single(
                X,
                n_clusters,
                size_min        = size_min,
                size_max        = size_max,
                max_iter        = max_iter,
                init            = init,
                verbose         = verbose,
                tol             = tol,
                x_squared_norms = x_squared_norms,
                random_state    = random_state,
                distance_metric = distance_metric)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels  = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter  = n_iter_
    else:
        # parallelisation of k-means runs
        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(kmeans_constrained_single)(X, n_clusters,
                                               size_min=size_min, size_max=size_max,
                                               max_iter=max_iter, init=init,
                                               verbose=verbose, tol=tol,
                                               x_squared_norms=x_squared_norms,
                                               # Change seed to ensure variety
                                               random_state=seed,
                                               distance_metric = distance_metric)
            for seed in seeds)
        # Get results with the lowest inertia
        labels, inertia, centers, n_iters = zip(*results)
        best = np.argmin(inertia)
        best_labels = labels[best]
        best_inertia = inertia[best]
        best_centers = centers[best]
        best_n_iter = n_iters[best]

    if not sp.issparse(X):
        if not copy_x:
            X += X_mean
        best_centers += X_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia


def k_means_constrained_two_data_sets(
                        x,
                        y,
                        distances,
                        n_clusters,
                        size_min        = None,
                        size_max        = None,
                        init            = 'k-means++',
                        n_init          = 10,
                        max_iter        = 300,
                        verbose         = False,
                        tolerance       = 1e-4,
                        random_state    = None,
                        copy_x          = True,
                        n_jobs          = 1,
                        return_n_iter   = False,
                        distance_metric = euclidean_distances,
                        **kwargs
                        ):
    """K-Means clustering with minimum and maximum cluster size constraints.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The observations to cluster.

    size_min : int, optional, default: None
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied

    size_max : int, optional, default: None
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    n_init : int, optional, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : boolean, optional
        Verbosity mode.

    tol : float, optional
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy_x : boolean, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    return_n_iter : bool, optional
        Whether or not to return the number of iterations.

    distance_metric : callable, default = euclidean_distances
        Distance metric used for calculating distance between points in the
        dataset. Defaults to the euclidean_distances function from
        sklearn.metrics.pairwise.
        If defining a custom function, it must accept two arguments: X, and
        cluster centers, in that order.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.

    """
    if sp.issparse(x):
        raise NotImplementedError("Not implemented for sparse x")
    if sp.issparse(y):
        raise NotImplementedError("Not implemented for sparse y")

    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
    random_state = check_random_state(random_state)

    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)

    x = as_float_array(x, copy=copy_x)
    # just commenting this out because it doesn't like my data
    # tolerance = _tolerance(x, tol)

    # Validate init array
    if hasattr(init, '__array__'):
        init = check_array(init, dtype=x.dtype.type, copy=True)
        _validate_center_shape(x, n_clusters, init)

        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # subtract of mean of x for more accurate distance computations
    # if not sp.issparse(x):
    #     x_mean = np.array(list(x.values())).mean(axis=0).item()
    #     # The copy was already done above
    #     x -= x_mean

    #     if hasattr(init, '__array__'):
    #         init -= x_mean

    # precompute squared norms of data points
    # x_squared_norms = row_norms(x, squared=True)

    best_labels, best_inertia, best_centers = None, None, None

    # For a single thread, less memory is needed if we just store one set
    # of the best results (as opposed to one set per run per thread).
    for it in range(n_init):
        # run a k-means once
        labels, inertia, centers, n_iter_ = kmeans_constrained_single_two_data_sets(
            x,
            y,
            distances,
            n_clusters,
            size_min        = size_min,
            size_max        = size_max,
            max_iter        = max_iter,
            init            = init,
            verbose         = verbose,
            tolerance       = tolerance,
            random_state    = random_state,
            distance_metric = distance_metric)
        # determine if these results are the best so far
        if best_inertia is None or inertia < best_inertia:
            best_labels  = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
            best_n_iter  = n_iter_

    if not sp.issparse(x):
        if not copy_x:
            x += x_mean
        best_centers += x_mean

    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_n_iter
    else:
        return best_centers, best_labels, best_inertia



def kmeans_constrained_single_two_data_sets(x,
                                            y,
                                            distances,
                                            n_clusters,
                                            size_min        = None,
                                            size_max        = None,
                                            max_iter        = 300,
                                            init            = 'k-means++',
                                            verbose         = False,
                                            random_state    = None,
                                            tolerance       = 1e-4,
                                            distance_metric = euclidean_distances,
                                            **kwargs):
    """A single run of k-means constrained, assumes preparation completed prior.

    Parameters
    ----------
    x : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    size_min : int, optional, default: None
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied

    size_max : int, optional, default: None
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments x, k and
        and a random state and return an initialization.

    tol : float, optional
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : boolean, optional
        Verbosity mode

    x_squared_norms : array
        Precomputed x_squared_norms.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    distance_metric : callable, default = euclidean_distances
        Distance metric used for calculating distance between points in the
        dataset. Defaults to the euclidean_distances function from
        sklearn.metrics.pairwise.
        If defining a custom function, it must accept two arguments: x, and
        cluster centers, in that order.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    if sp.issparse(x):
        raise NotImplementedError("Not implemented for sparse x")

    random_state = check_random_state(random_state)
    # n_samples = x.shape[0]

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids_two_data_sets(x, y, distances, n_clusters, init, random_state=random_state)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(len(x),len(y)))

    # Determine min and max sizes if non given
    if size_min is None:
        size_min = 0
    if size_max is None:
        size_max = n_samples  # Number of data points

    # Check size min and max
    # if not ((size_min >= 0) and (size_min <= n_samples)
    #         and (size_max >= 0) and (size_max <= n_samples)):
    #     raise ValueError("size_min and size_max must be a positive number smaller "
    #                      "than the number of data points or `None`")
    # if size_max < size_min:
    #     raise ValueError("size_max must be larger than size_min")
    # if size_min * n_clusters > n_samples:
    #     raise ValueError("The product of size_min and n_clusters cannot exceed the number of samples (x)")
    # if size_max * n_clusters < n_samples:
    #     raise ValueError("The product of size_max and n_clusters must be larger than or equal the number of samples (x)")

    # iterations
    # the `i` variable is for printing, and it's also returned
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_constrained_two_data_sets(
                                x,
                                y,
                                centers,
                                size_min,
                                size_max,
                                distances       = distances,
                                distance_metric = distance_metric,
                                **kwargs)

        # computation of the means is also called the M-step of EM
        centers = _centers_dense_two_data_sets(x, y, labels, n_clusters, distances)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels  = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tolerance:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tolerance))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_constrained_two_data_sets(x,
                                centers,
                                size_min,
                                size_max,
                                distances       = distances,
                                distance_metric = distance_metric,
                                **kwargs)

    return best_labels, best_inertia, best_centers, i + 1

def kmeans_constrained_single(X, n_clusters, size_min=None, size_max=None,
                              max_iter=300, init='k-means++',
                              verbose=False, x_squared_norms=None,
                              random_state=None, tol=1e-4,
                              distance_metric=euclidean_distances, **kwargs):
    """A single run of k-means constrained, assumes preparation completed prior.

    Parameters
    ----------
    X : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    size_min : int, optional, default: None
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied

    size_max : int, optional, default: None
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': generate k centroids from a Gaussian with mean and
        variance estimated from the data.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    tol : float, optional
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : boolean, optional
        Verbosity mode

    x_squared_norms : array
        Precomputed x_squared_norms.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    distance_metric : callable, default = euclidean_distances
        Distance metric used for calculating distance between points in the
        dataset. Defaults to the euclidean_distances function from
        sklearn.metrics.pairwise.
        If defining a custom function, it must accept two arguments: X, and
        cluster centers, in that order.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    if sp.issparse(X):
        raise NotImplementedError("Not implemented for sparse X")

    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(n_samples,), dtype=X.dtype)

    # Determine min and max sizes if non given
    if size_min is None:
        size_min = 0
    if size_max is None:
        size_max = n_samples  # Number of data points

    # Check size min and max
    if not ((size_min >= 0) and (size_min <= n_samples)
            and (size_max >= 0) and (size_max <= n_samples)):
        raise ValueError("size_min and size_max must be a positive number smaller "
                         "than the number of data points or `None`")
    if size_max < size_min:
        raise ValueError("size_max must be larger than size_min")
    if size_min * n_clusters > n_samples:
        raise ValueError("The product of size_min and n_clusters cannot exceed the number of samples (X)")
    if size_max * n_clusters < n_samples:
        raise ValueError("The product of size_max and n_clusters must be larger than or equal the number of samples (X)")

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_constrained(X, centers, size_min, size_max, distances=distances,
                                distance_metric = distance_metric, **kwargs)

        # computation of the means is also called the M-step of EM
        if sp.issparse(X):
            centers = _centers_sparse(X, labels, n_clusters, distances)
        else:
            centers = _centers_dense(X, labels, n_clusters, distances)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_constrained(X, centers, size_min, size_max, distances=distances,
                                 distance_metric = distance_metric, **kwargs)

    return best_labels, best_inertia, best_centers, i + 1


def _labels_constrained(X, centers, size_min, size_max, distances, distance_metric, **kwargs ):
    """Compute labels using the min and max cluster size constraint

    This will overwrite the 'distances' array in-place.

    Parameters
    ----------
    X : numpy array, shape (n_sample, n_features)
        Input data.

    size_min : int
        Minimum size for each cluster

    size_max : int
        Maximum size for each cluster

    centers : numpy array, shape (n_clusters, n_features)
        Cluster centers which data is assigned to.

    distances : numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.

    distance_metric : Callable
        Function which calculates distances between points

    Returns
    -------
    labels : numpy array, dtype=np.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.

    """
    C = centers

    # Distances to each centre C. (the `distances` parameter is the distance to the closest centre)
    # K-mean original uses squared distances but this equivalent for constrained k-means
    D = distance_metric(X, C, **kwargs)

    edges, costs, capacities, supplies, n_C, n_X = minimum_cost_flow_problem_graph(X, C, D, size_min, size_max)
    labels = solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X)

    # cython k-means M step code assumes int32 inputs
    labels = labels.astype(np.int32)

    # Change distances in-place
    # `arange` isn't "arrange", it's "A range" as in an array containing a range of integers
    distances[:] = D[np.arange(D.shape[0]), labels] ** 2  # Square for M step of EM
    inertia = distances.sum()

    return labels, inertia


def _labels_constrained_two_data_sets(x, y, centers, size_min, size_max, distances, distance_metric, **kwargs ):
    """compute labels using the min and max cluster size constraint

    This will overwrite the 'distances' array in-place.

    Parameters
    ----------
    x : numpy array, shape (n_sample, n_features)
        Input data.

    size_min : int
        Minimum size for each cluster

    size_max : int
        Maximum size for each cluster

    centers : numpy array, shape (n_clusters, n_features)
        cluster centers which data is assigned to.

    distances : numpy array, shape (n_samples,)
        Pre-allocated array in which distances are stored.

    distance_metric : callable
        Function which calculates distances between points

    Returns
    -------
    labels : numpy array, dtype=np.int, shape (n_samples,)
        Indices of clusters that samples are assigned to.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.

    """
    c = centers

    # distances to each centre c. (the `distances` parameter is the distance to the closest centre)
    # K-mean original uses squared distances but this equivalent for constrained k-means
    d = distance_metric(x, c, **kwargs)

    edges, costs, capacities, supplies, n_c, n_x = minimum_cost_flow_problem_graph_two_data_sets(x, y, c, d, size_min, size_max)
    labels = solve_min_cost_flow_graph_two_data_sets(edges, costs, capacities, supplies, n_c, n_x)

    # cython k-means M step code assumes int32 inputs
    labels = labels.astype(np.int32)

    # I don't think we need this squaring stuff because we should already have the distances
    #
    # change distances in-place
    # `arange` isn't "arrange", it's "A range" as in an array containing a range of integers
    # Expectation-Maximization (EM) algorithm, specifically the "M step" (Maximization step) of the algorithm.
    # distances[:] = d[np.arange(d.shape[0]), labels] ** 2  # Square for M step of EM
    inertia = distances.sum()

    return labels, inertia



def minimum_cost_flow_problem_graph_two_data_sets(cabinets, clients, centers, d, size_min, size_max):
    # Setup minimum cost flow formulation graph
    # Vertices indexes:
    # cabinets-nodes: [0, n(cabinets)-1], c-nodes: [n(cabinets), n(cabinets)+n(c)-1], c-dummy nodes:[n(cabinets)+n(c), n(cabinets)+2*n(c)-1],
    # Artificial node: [n(cabinets)+2*n(c), n(cabinets)+2*n(c)+1-1]

    # create indices of nodes
    cabinets_length      = len(cabinets)              # the length of the cabinets data
    centers_length       = len(centers)               # the length of the c data
    cabinets_index_range = np.arange(cabinets_length) # an array containing the index of every element in the cabinets data

    centers_dummy_index_range = np.arange(cabinets_index_range[-1]      + 1, cabinets_index_range[-1]      + 1 + centers_length) # an array containing indexes starting at the length of the cabinets data and ending at the length of the cabinets data plus the length of the c data
    centers_index_range       = np.arange(centers_dummy_index_range[-1] + 1, centers_dummy_index_range[-1] + 1 + centers_length) # an array containing indexes starting at the length of the total of the cabinets data and c data and ending at that number plus the length of the c data
    artificial_node_index     = centers_index_range[-1]                 + 1                                                      # the length of the previous array

    # Edges
    edges_cabinets_centers_dummy = cartesian([cabinets_index_range, centers_dummy_index_range])                              # All cabinets connect to all c dummy nodes (c')
    edges_centers_dummy_centers  = np.stack([centers_dummy_index_range, centers_index_range],                       axis=1)  # Each c connects to a corresponding c (centroid)
    edges_centers_artificial     = np.stack([centers_index_range, artificial_node_index * np.ones(centers_length)], axis=1)  # All c connect to artificial node

    edges = np.concatenate([edges_cabinets_centers_dummy, edges_centers_dummy_centers, edges_centers_artificial])

    # costs
    #
    # `.size` gets the total number of deep elements, instead of just getting the size of the array
    # this line essentially just makes it a one dimensional array
    # this isn't gonna work with our two set data
    # we need to keep the two sets distinct
    costs_centers_dummy = d.reshape(d.size)
    # costs               = np.concatenate([costs_centers_dummy, np.zeros(edges.shape[0] - len(costs_centers_dummy))])
    # costs               = np.concatenate([costs_centers_dummy, np.zeros(edges.shape[0])])
    # hard coding for now until I figure out what's going on
    costs               = np.concatenate([np.zeros(6171)])

    # capacities - can set for max-k
    capacities_centers_dummy_centers = size_max * np.ones(centers_length)
    capacities_nonrestricted         = cabinets_length  # The total supply and therefore won't restrict flow
    capacities                       = np.concatenate([
        np.ones(edges_cabinets_centers_dummy.shape[0]),
        capacities_centers_dummy_centers,
        capacities_nonrestricted * np.ones(centers_length)
    ])

    # Sources and sinks
    supplies_cabinets   = np.ones(cabinets_length)
    supplies_centers    = -1 * size_min * np.ones(centers_length)            # demand node
    supplies_artificial = -1 * (cabinets_length - centers_length * size_min) # demand node
    supplies            = np.concatenate([
        supplies_cabinets,
        np.zeros(centers_length), # c_dummies
        supplies_centers,
        [supplies_artificial]
    ])

    # All arrays must be of int dtype for `SimpleMincostFlow`
    edges      = edges.astype('int32')
    # are you kidding me, this is array round, not around. this method rounds the elements of the array
    # it doesn't have anything to do with something around the elements
    costs      = np.around(costs * 1000, 0).astype('int32')  # Times by 1000 to give ecabinetstra precision
    capacities = capacities.astype('int32')
    supplies   = supplies.astype('int32')

    return edges, costs, capacities, supplies, centers_length, cabinets_length

def minimum_cost_flow_problem_graph(x, centers, d, size_min, size_max):
    # Setup minimum cost flow formulation graph
    # Vertices indexes:
    # x-nodes: [0, n(x)-1], c-nodes: [n(x), n(x)+n(c)-1], c-dummy nodes:[n(x)+n(c), n(x)+2*n(c)-1],
    # Artificial node: [n(x)+2*n(c), n(x)+2*n(c)+1-1]

    # create indices of nodes
    x_length       = x.shape[0]          # the length of the x data
    centers_length = centers.shape[0]    # the length of the c data
    x_index_range  = np.arange(x_length) # an array containing the indexes of every element in the x data

    centers_dummy_index_range = np.arange(x_index_range[-1]             + 1, x_index_range[-1]             + 1 + centers_length) # an array containing indexes starting at the length of the x data and ending at the length of the x data plus the length of the c data
    centers_index_range       = np.arange(centers_dummy_index_range[-1] + 1, centers_dummy_index_range[-1] + 1 + centers_length) # an array containing indexes starting at the length of the total of the x data and c data and ending at that number plus the length of the c data
    artificial_node_index     = centers_index_range[-1] + 1                                                                      # the length of the previous array

    # Edges
    edges_x_centers_dummy       = cartesian([x_index_range, centers_dummy_index_range])                                     # All x's connect to all c dummy nodes (c')
    edges_centers_dummy_centers = np.stack([centers_dummy_index_range, centers_index_range],                       axis=1)  # Each c' connects to a corresponding c (centroid)
    edges_centers_artificial    = np.stack([centers_index_range, artificial_node_index * np.ones(centers_length)], axis=1)  # All c connect to artificial node

    edges = np.concatenate([edges_x_centers_dummy, edges_centers_dummy_centers, edges_centers_artificial])

    # costs
    #
    # `.size` gets the total number of deep elements, instead of just getting the size of the array
    # this line essentially just makes it a one dimensional array
    # this isn't gonna work with our two set data
    # we need to keep the two sets distinct
    costs_x_centers_dummy = d.reshape(d.size)
    costs                 = np.concatenate([costs_x_centers_dummy, np.zeros(edges.shape[0] - len(costs_x_centers_dummy))])

    # capacities - can set for max-k
    capacities_centers_dummy_centers = size_max * np.ones(centers_length)
    capacities_non                   = x_length  # The total supply and therefore wont restrict flow
    capacities                       = np.concatenate([
        np.ones(edges_x_centers_dummy.shape[0]),
        capacities_centers_dummy_centers,
        capacities_non * np.ones(centers_length)
    ])

    # Sources and sinks
    supplies_x          = np.ones(x_length)
    supplies_centers    = -1 * size_min * np.ones(centers_length)     # demand node
    supplies_artificial = -1 * (x_length - centers_length * size_min) # demand node
    supplies            = np.concatenate([
        supplies_x,
        np.zeros(centers_length), # c_dummies
        supplies_centers,
        [supplies_artificial]
    ])

    # All arrays must be of int dtype for `SimpleMincostFlow`
    edges      = edges.astype('int32')
    # are you kidding me, this is array round, not around. this method rounds the elements of the array
    # it doesn't have anything to do with something around the elements
    costs      = np.around(costs * 1000, 0).astype('int32')  # Times by 1000 to give extra precision
    capacities = capacities.astype('int32')
    supplies   = supplies.astype('int32')

    return edges, costs, capacities, supplies, centers_length, x_length

def solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = SimpleMinCostFlow()

    if (edges.dtype != 'int32') or (costs.dtype != 'int32') \
            or (capacities.dtype != 'int32') or (supplies.dtype != 'int32'):
        raise ValueError("`edges`, `costs`, `capacities`, `supplies` must all be int dtype")

    N_edges = edges.shape[0]
    N_nodes = len(supplies)

    # Add each edge with associated capacities and cost
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(edges[:, 0], edges[:, 1], capacities, costs)

    # Add node supplies
    min_cost_flow.set_nodes_supplies(np.arange(len(supplies)), supplies)

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.solve() != min_cost_flow.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    # Assignment
    labels_M = np.array([min_cost_flow.flow(i) for i in range(n_X * n_C)]).reshape(n_X, n_C).astype('int32')

    labels = labels_M.argmax(axis=1)
    return labels

def solve_min_cost_flow_graph_two_data_sets(edges, costs, capacities, supplies, n_c, n_x):
    # Instantiate a SimplemincostFlow solver.
    min_cost_flow = SimpleMinCostFlow()

    if (edges.dtype != 'int32') or (costs.dtype != 'int32') \
            or (capacities.dtype != 'int32') or (supplies.dtype != 'int32'):
        raise ValueError("`edges`, `costs`, `capacities`, `supplies` must all be int dtype")

    n_edges = edges.shape[0]
    n_nodes = len(supplies)

    # Add each edge with associated capacities and cost
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(edges[:, 0], edges[:, 1], capacities, costs)

    # Add node supplies
    min_cost_flow.set_nodes_supplies(np.arange(len(supplies)), supplies)

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.solve() != min_cost_flow.OPTIMAL:
        raise Exception('There was an issue with the min cost flow input.')

    # Assignment
    labels_m = np.array([min_cost_flow.flow(i) for i in range(n_x * n_c)]).reshape(n_x, n_c).astype('int32')

    # `argmax` returns the indices of the maximum values along a specified axis in an array. In other words, it gives you the position of the maximum element in a NumPy array.
    labels = labels_m.argmax(axis=1)
    return labels


class KMeansConstrained(KMeans):
    """K-Means clustering with minimum and maximum cluster size constraints

    Parameters
    ----------

    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    size_min : int, optional, default: None
        Constrain the label assignment so that each cluster has a minimum
        size of size_min. If None, no constrains will be applied

    size_max : int, optional, default: None
        Constrain the label assignment so that each cluster has a maximum
        size of size_max. If None, no constrains will be applied

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    n_init : int, default: 10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default: 1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default 0
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    distance_metric : callable, default = euclidean_distances
        Distance metric used for calculating distance between points in the
        dataset. Defaults to the euclidean_distances function from
        sklearn.metrics.pairwise.
        If defining a custom function, it must accept two arguments: X, and
        cluster centers, in that order.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.

    Examples
    --------

    >>> from k_means_constrained import KMeansConstrained
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...                [4, 2], [4, 4], [4, 0]])
    >>> clf = KMeansConstrained(
    ...     n_clusters=2,
    ...     size_min=2,
    ...     size_max=5,
    ...     random_state=0
    ... )
    >>> clf.fit_predict(X)
    array([0, 0, 0, 1, 1, 1], dtype=int32)
    >>> clf.cluster_centers_
    array([[ 1.,  2.],
           [ 4.,  2.]])
    >>> clf.labels_
    array([0, 0, 0, 1, 1, 1], dtype=int32)

    Notes
    ------
    K-means problem constrained with a minimum and/or maximum size for each cluster.

    The constrained assignment is formulated as a Minimum Cost Flow (MCF) linear network optimisation
    problem. This is then solved using a cost-scaling push-relabel algorithm. The implementation used is
     Google's Operations Research tools's `SimpleMinCostFlow`.

    Ref:
    1. Bradley, P. S., K. P. Bennett, and Ayhan Demiriz. "Constrained k-means clustering."
        Microsoft Research, Redmond (2000): 1-8.
    2. Google's SimpleMinCostFlow implementation:
        https://github.com/google/or-tools/blob/master/ortools/graph/min_cost_flow.h
    """

    def __init__(self, n_clusters=8, size_min=None, size_max=None, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 verbose=False, random_state=None, copy_x=True, n_jobs=1, distance_metric = euclidean_distances):

        self.size_min        = size_min
        self.size_max        = size_max
        self.distance_metric = distance_metric

        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                         verbose=verbose, random_state=random_state, copy_x=copy_x, n_jobs=n_jobs)

    def fit(self, X, y=None):
        """Compute k-means clustering with given constants.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored

        """
        if sp.issparse(X):
            raise NotImplementedError("Not implemented for sparse X")

        random_state = check_random_state(self.random_state)
        X            = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means_constrained(
                X, n_clusters=self.n_clusters,
                size_min=self.size_min, size_max=self.size_max,
                init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs,
                return_n_iter=True, distance_metric=self.distance_metric)
        return self



    def fit_two_data_sets(self, x, y=None):
        """Compute k-means clustering with given constants.

        Parameters
        ----------
        x : array-like, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored

        """
        if sp.issparse(x):
            raise NotImplementedError("Not implemented for sparse x")

        random_state = check_random_state(self.random_state)
        x            = self._check_fit_data(x)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means_constrained_two_data_sets(
                x,
                y,
                n_clusters      = self.n_clusters,
                size_min        = self.size_min,
                size_max        = self.size_max,
                init            = self.init,
                n_init          = self.n_init,
                max_iter        = self.max_iter,
                verbose         = self.verbose,
                tol             = self.tol,
                random_state    = random_state,
                copy_x          = self.copy_x,
                n_jobs          = self.n_jobs,
                return_n_iter   = True,
                distance_metric = self.distance_metric)
        return self


    def fit_two_data_sets(self, x, y, distances):
        """Compute k-means clustering with given constants.

        Parameters
        ----------
        x : array-like, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored

        """
        if sp.issparse(x):
            raise NotImplementedError("Not implemented for sparse x")

        random_state = check_random_state(self.random_state)
        x            = self._check_fit_data(x)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means_constrained_two_data_sets(
                x,
                y,
                distances,
                n_clusters     = self.n_clusters,
                size_min       = self.size_min,
                size_max       = self.size_max,
                init           = self.init,
                n_init         = self.n_init,
                max_iter       = self.max_iter,
                verbose        = self.verbose,
                tol            = self.tol,
                random_state   = random_state,
                copy_x         = self.copy_x,
                n_jobs         = self.n_jobs,
                return_n_iter  = True,
                distance_metric= self.distance_metric
            )
        return self

    def predict(self, X, size_min='init', size_max='init'):
        """
        Predict the closest cluster each sample in X belongs to given the provided constraints.
        The constraints can be temporally overridden when determining which cluster each datapoint is assigned to.

        Only computes the assignment step. It does not re-fit the cluster positions.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.

        size_min : int, optional, default: size_min provided with initialisation
            Constrain the label assignment so that each cluster has a minimum
            size of size_min. If None, no constrains will be applied.
            If 'init' the value provided during initialisation of the
            class will be used.

        size_max : int, optional, default: size_max provided with initialisation
            Constrain the label assignment so that each cluster has a maximum
            size of size_max. If None, no constrains will be applied.
            If 'init' the value provided during initialisation of the
            class will be used.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """

        if sp.issparse(X):
            raise NotImplementedError("Not implemented for sparse X")

        if size_min == 'init':
            size_min = self.size_min
        if size_max == 'init':
            size_max = self.size_max

        n_clusters = self.n_clusters
        n_samples = X.shape[0]

        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)

        # Allocate memory to store the distances for each sample to its
        # closer center for reallocation in case of ties
        distances = np.zeros(shape=(n_samples,), dtype=X.dtype)

        # Determine min and max sizes if non given
        if size_min is None:
            size_min = 0
        if size_max is None:
            size_max = n_samples  # Number of data points

        # Check size min and max
        if not ((size_min >= 0) and (size_min <= n_samples)
                and (size_max >= 0) and (size_max <= n_samples)):
            raise ValueError("size_min and size_max must be a positive number smaller "
                             "than the number of data points or `None`")
        if size_max < size_min:
            raise ValueError("size_max must be larger than size_min")
        if size_min * n_clusters > n_samples:
            raise ValueError("The product of size_min and n_clusters cannot exceed the number of samples (X)")

        labels, inertia = \
            _labels_constrained(X, self.cluster_centers_, size_min, size_max, distances=distances, distance_metric=self.distance_metric)

        return labels

    def fit_predict(self, X, y=None):
        """Compute cluster centers and predict cluster index for each sample.

        Equivalent to calling fit(X) followed by predict(X) but also more efficient.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

    def fit_predict_two_data_sets(self, x, y, distances):
        return self.fit_two_data_sets(x, y, distances).labels_