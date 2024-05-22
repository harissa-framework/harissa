import numpy as np

from scipy.special import gammaln

def log_gamma_poisson_pdf(k, a, b):

    ga, gk = gammaln(a), gammaln(k + 1)
    return gammaln(a + k) - (ga + gk) + a * np.log(b) - (a + k) * np.log(b + 1)

# def estim_gamma_poisson(x):
#     """
#     Estimate parameters a and b of the Gamma-Poisson(a,b) distribution,
#     a.k.a. negative binomial distribution, using the method of moments.
#     """
#     m1 = np.mean(x)
#     m2 = np.mean(x*(x-1))
#     if m1 == 0:
#         return 0, 1
#     r = m2 - m1**2
#     if r > 0:
#         a = m1 ** 2 / r
#     else:
#         v = np.var(x)
#         if v == 0:
#             return 0, 1
#         a = m1 ** 2 /v
#     b = a / m1
#     return a, b

def core_basins_binary( 
    counts_gene,
    data_bool, 
    burst_frequencies, 
    burst_inv_size, 
    weight
):
    """
    Compute the basal parameters for a gene.
    """
    for cell, count in enumerate(counts_gene):
        for i, burst_frequency in enumerate(burst_frequencies):
            weight[cell, i] = log_gamma_poisson_pdf(
                count, 
                burst_frequency, 
                burst_inv_size
            )
        data_bool[cell] = burst_frequencies[np.argmax(weight[cell, :])]


# def build_cnt(cnt, cnt_move, vect_kon, vect_t, times, n_genes_stim, p):
#     for j in range(1, n_genes_stim):
#         for i in range(1, n_genes_stim):
#             tmp_cnt = 0
#             if cnt > 2: 
#             # -2 for letting the first time points after the stimulus unchanged
#                 for t in range(0, cnt - 2):
#                     tmp = np.sum(vect_t == times[t])
#                     tmp1 = np.sum(vect_kon[vect_t == times[t], i] == 1) / tmp
#                     tmp2 = np.sum(vect_kon[vect_t == times[t], j] == 1) / tmp
#                     if tmp1 > p:
#                         tmp_cnt += tmp1 * tmp2
#                 tmp_cnt /= cnt-1
#             cnt_move[j, i] = (1 + tmp_cnt) * max(
#                 p, 
#                 min(
#                     1 
#                     + np.sum(vect_kon[vect_t == times[cnt - 2], i] == 1) 
#                     / np.sum(vect_t == times[cnt - 2]) 
#                     - np.sum(vect_kon[vect_t == times[cnt - 1], i] == 1) 
#                     / np.sum(vect_t == times[cnt - 1]),
#                     1
#                 )
#             )
#     return cnt_move