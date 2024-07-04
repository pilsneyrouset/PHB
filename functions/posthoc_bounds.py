import numpy as np
import matplotlib.pyplot as plt
import sanssouci


def interpolation_naif(p_values: list, thresholds: list, zeta: list=[k for k in range(5000)]) -> int:
    "Naive calculus posthoc bounds"
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


def interpolation(p_values: list, thresholds: list, zeta: list=[k for k in range(5000)]) -> int:
    "Calcul posthoc bounds"
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


def interpolation_bis(p_values: list, thresholds:list, zeta: list=[k for k in range(5000)]) -> int:
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


def interpolation_minmax_naif(p_values: list, thresholds: list, kmin:int, kmax: int, zeta: list=[k for k in range(5000)]) -> int:
    "Naive calculus of the posthoc bounds with kmin and kmax"
    p_values = np.sort(p_values)
    B = []
    if kmin == kmax:
        return len(p_values)
    else:
        for k in range(kmin, kmax):
            compteur = 0
            for p_value in p_values:
                if p_value >= thresholds[k]:
                    compteur += 1
            B.append(compteur + zeta[k])
        return min(min(B), len(p_values))


def interpolation_minmax(p_values: list, thresholds: list, kmin: int, kmax: int, zeta: list= [k for k in range(5000)]) -> int:
    "La bound posthoc est calculée pour les k compris entre kmin et kmax"
    p_values = np.sort(p_values)
    B = []
    i_start = 0
    if kmin == kmax:
        i = 0
        while i < len(p_values):
            if p_values[i] < thresholds[kmax - 1]:
                i += 1
            else:
                return min(len(p_values) - i + zeta[kmax - 1], len(p_values))
    else:
        for k in range(kmin, min(kmax, len(p_values))):
            i = i_start
            while i < len(p_values) and p_values[i] < thresholds[k]:
                    i += 1
                    i_start = i
            B.append(len(p_values) - i + zeta[k])
        return min(min(B), len(p_values))


def linear_interpolation(p_values : list, thresholds: list, kmin: int=0)-> list:
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


def linear_interpolation_zeta(p_values: list, thresholds: list, zeta, kmin: int=0) -> list:
    p_values = np.sort(p_values)
    s = len(p_values)
    K = len(thresholds)
    tau = np.zeros(K)
    K = max(s, K)
    for k in range(K):
        if k < kmin:
            tau[k] = 0
        elif kmin <= k < len(thresholds):
            tau[k] = thresholds[k]
        else:
            tau[k] = thresholds[len(thresholds)-1]
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
    V, A, M = np.zeros(s), np.zeros(K), np.zeros(K)
    M[0] = r[0]
    for k in range(K):
        A[k] = r[k] - zeta[k]
        if k > 0:
            M[k] = max(M[k-1], A[k])
    for i in range(s):
        if kappa[i] > 0:
            V[i] = int(min(zeta[kappa[i]], i+1 - M[kappa[i]-1]))
    return V