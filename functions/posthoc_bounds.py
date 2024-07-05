import numpy as np
import matplotlib.pyplot as plt
import warnings


def interpolation_naif(p_values: list, thresholds: list, zeta: list) -> int:
    """
    Naive method to compute an upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    int:
        An upper bound for the number of false discoveries among
        the given items
    """
    p_values = np.sort(p_values)
    K = len(thresholds)
    kmax = min(len(p_values), K)
    B = []
    for k in range(kmax):
        compteur = 0
        for p_value in p_values:
            if p_value >= thresholds[k]:
                compteur += 1
        B.append(compteur + zeta[k])
    return min(min(B), len(p_values))


def interpolation(p_values: list, thresholds: list, zeta: list) -> int:
    """
    Upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    int:
        An upper bound for the number of false discoveries among
        the given items
    """
    p_values = np.sort(p_values)
    K = len(thresholds)
    kmax = min(len(p_values), K)
    B = []
    i_start = 0
    for k in range(kmax):
        i = i_start
        while i < len(p_values) and p_values[i] < thresholds[k]:
            i += 1
            i_start = i
        B.append(len(p_values) - i + zeta[k])
    return min(min(B), len(p_values))


def interpolation_bis(p_values: list, thresholds:list, zeta: list) -> int:
    """
    Upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
        thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    int :
        An upper bound for the number of false discoveries among
        the given p-values
    """
    p_values = np.sort(p_values)
    s = len(p_values)
    K = len(thresholds)
    B = []
    k, i = 0, 0
    while k < K and i < s:
        if p_values[i] < thresholds[k]:
            i += 1
        else:
            B.append(s - i + zeta[k])
            k += 1
    B.append(s - i + zeta[k])
    return min(min(B), len(p_values))


def interpolation_minmax_naif(p_values: list, thresholds: list, kmin:int, kmax: int, zeta: list) -> int:
    """
    Naive method to compute and upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
        thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    int :
        An upper bound for the number of false discoveries among
        the given p-values for the indexes between kmin and kmax
    """
    p_values = np.sort(p_values)
    B = []
    if kmin == kmax:
        warnings.warn("Attention, la liste de p-valeurs est vide. La borne triviale est renvoyée")
        return len(p_values)
    else:
        for k in range(kmin, kmax):
            compteur = 0
            for p_value in p_values:
                if p_value >= thresholds[k]:
                    compteur += 1
            B.append(compteur + zeta[k])
        return min(min(B), len(p_values))


def interpolation_minmax(p_values: list, thresholds: list, kmin: int, kmax: int, zeta: list) -> int:
    """
    Upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    int:
        An upper bound for the number of false discoveries among
        the given p-values for the indexes between kmin and kmax
    """
    p_values = np.sort(p_values)
    B = []
    i_start = 0
    if kmin == kmax:
       warnings.warn("Attention, la liste de p-valeurs est vide. La borne triviale est renvoyée")
       return len(p_values)
    else:
        for k in range(kmin, kmax):
            i = i_start
            while i < len(p_values) and p_values[i] < thresholds[k]:
                    i += 1
                    i_start = i
            B.append(len(p_values) - i + zeta[k])
        return min(min(B), len(p_values))


def linear_interpolation(p_values : list, thresholds: list, kmin: int=0)-> list:
    """
    Upper bound for the number of false discoveries among most
    significant items in the case zeta_k=k-1.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    numpy.array :
        A vector of size p giving an joint upper confidence bound on the
        number of false discoveries among the k most significant items for
        all k in \{1,\ldots,m\}

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.    
    """

    p_values = np.sort(p_values)
    s = len(p_values)
    K = len(thresholds)
    tau = np.zeros(s)
    for k in range(s):
        if k < kmin:
            tau[k] = 0
        elif kmin <= k < K:
            tau[k] = thresholds[k]
        else:
            tau[k] = thresholds[K-1]
    kappa = np.ones(s, dtype=int)*s
    r = np.ones(s, dtype=int)*s
    k, i = 0, 0
    while k < s and i < s:
        if p_values[i] < tau[k]:
            kappa[i] = k
            i += 1
        else:
            r[k] = i
            k += 1
    V, A, M = np.zeros(s), np.zeros(s), np.zeros(s)
    for k in range(s):
        A[k] = r[k] - k
        if k > 0:
            M[k] = max(M[k-1], A[k])
    for i in range(s):
        if kappa[i] > 0:
            V[i] = int(min(kappa[i], i+1 - M[kappa[i]-1]))
    return V


def linear_interpolation_zeta(p_values: list, thresholds: list, zeta: list, kmin: int=0) -> list:
    """
    Upper bound for the number of false discoveries among most
    significant items in the general case.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    -------

    numpy.array :
        A vector of size p giving an joint upper confidence bound on the
        number of false discoveries among the k most significant items for
        all k in \{1,\ldots,m\}

    References
    ----------

    .. [1] Blanchard, G., Neuvial, P., & Roquain, E. (2020). Post hoc
        confidence bounds on false positives using reference families.
        Annals of Statistics, 48(3), 1281-1303.
    """
    p_values = np.sort(p_values)
    s = len(p_values)
    K = len(thresholds)
    K = max(s, K)
    tau = np.zeros(K)
    ksi = np.zeros(K)
    for k in range(K):
        if k < kmin:
            tau[k] = 0
            ksi[k] = zeta[k]
        elif kmin <= k < len(thresholds):
            tau[k] = thresholds[k]
            ksi[k] = zeta[k]
        else:
            tau[k] = thresholds[len(thresholds)-1]
            ksi[k] = zeta[len(thresholds)-1]
    kappa = np.ones(K, dtype=int)*K
    r = np.ones(K, dtype=int)*s
    k, i = 0, 0
    while k < K and i < s:
        if p_values[i] < tau[k]:
            kappa[i] = k
            i += 1
        else:
            r[k] = i
            k += 1
    V, A, M = np.ones(s)*ksi[0], np.zeros(K), np.zeros(K)
    M[0] = max(r[0] - ksi[0], 0)
    for k in range(K):
        A[k] = r[k] - ksi[k]
        if k > 0:
            M[k] = max(M[k-1], A[k])
    for i in range(s):
        if kappa[i] > 0:
            if kappa[i] != K:
                V[i] = int(min(ksi[kappa[i]], i+1 - M[kappa[i]-1]))
            else:
                V[i] = int(i+1 - M[kappa[i]-1])
    return V


def argmin_indice(p_values: list, thresholds: list, kmin: int, kmax: int, zeta: list) -> int:
    """Upper bound for the number of false discoveries among most
    significant items.

    Parameters
    ----------

    p_values : 1D numpy.array
        A 1D numpy array containing all p-values,sorted non-decreasingly
    thresholds : 1D numpy.array
        A 1D numpy array  of K JER-controlling thresholds,
        sorted non-decreasingly

    Returns
    """
    p_values = np.sort(p_values)
    B = []
    i_start = 0
    if kmin == kmax:
       warnings.warn('Attention, la liste de p-valeurs est vide')
       return -1
    else:
        for k in range(kmin, kmax):
            i = i_start
            while i < len(p_values) and p_values[i] < thresholds[k]:
                    i += 1
                    i_start = i
            B.append(len(p_values) - i + zeta[k])
        if min(B) < len(p_values):
            return np.argmin(B)
        else:
            warnings.warn("Nous avons la borne triviale, l'indice renvoyé est -1")
            return -1