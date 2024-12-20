import pandas as pd
import numpy as np
from numba import njit, prange
from scipy.spatial.distance import squareform, cdist, euclidean

from .general_functions import def_var_value_if_none


class IntegrationAprox:
    def __init__(self, f_val_at_x, x_val=None):
        self.x_val = def_var_value_if_none(value_passed=x_val, default=list(range(len(f_val_at_x))))
        self.fanc_val_at_x = f_val_at_x
        self.space_change = self.find_equally_spaced_regions(self.x_val)
        self.res_area = self.integration_composite_rule()

    @staticmethod
    def find_equally_spaced_regions(x_val):
        x_val_spaces = []
        for i in range(len(x_val) - 1):
            x_val_spaces += [np.round(x_val[i + 1] - x_val[i], 6)]

        space_change = [0]
        for i in range(len(x_val_spaces) - 1):
            if not x_val_spaces[i] == x_val_spaces[i + 1]:
                space_change += [i + 1]

        space_change += [len(x_val)]
        return space_change

    def integration_composite_rule(self):
        if not self.space_change:
            return self.uniform_spaced_integration_comp_rule(self.x_val)

        uni_space_sum = 0
        for i in range(len(self.space_change) - 1):
            uni_space_sum += self.uniform_spaced_integration_comp_rule(
                self.x_val[self.space_change[i]:self.space_change[i + 1] + 1])

        return uni_space_sum

    def uniform_spaced_integration_comp_rule(self, x_val):
        comp_sum = 0
        if len(x_val) == 1:
            print("This should be impossible!")
        elif len(x_val) == 2:
            h_len = x_val[1] - x_val[0]
            poss = self.x_val.index(x_val[0])
            comp_sum += self.integral_trapezoid_type(self.fanc_val_at_x[poss:poss + 2], h_len=h_len)
        elif len(x_val) == 3:
            h_len = x_val[1] - x_val[0]
            poss = self.x_val.index(x_val[0])
            comp_sum += self.integral_simpson_1_3(self.fanc_val_at_x[poss:poss + 3], h_len=h_len)
        elif len(x_val) == 4:
            h_len = x_val[1] - x_val[0]
            poss = self.x_val.index(x_val[0])
            comp_sum += self.integral_simpson_3_8(self.fanc_val_at_x[poss:poss + 4], h_len=h_len)
        elif len(x_val) > 4:
            if len(x_val) % 4 == 3 or len(x_val) % 4 == 2 or len(x_val) % 4 == 0:
                comp_sum += self.uniform_spaced_integration_comp_rule(x_val[:4])
                comp_sum += self.uniform_spaced_integration_comp_rule(x_val[3:])
            elif len(x_val) % 4 == 1:
                comp_sum += self.uniform_spaced_integration_comp_rule(x_val[:3])
                comp_sum += self.uniform_spaced_integration_comp_rule(x_val[2:])

        return comp_sum

    @classmethod
    def trapezoid_comp_rule(cls, x_val: (list, np.ndarray)):
        return x_val[0]/2 + sum(x_val[1:-1]) + x_val[-1]/2

    @staticmethod
    def integral_trapezoid_type(func_values, h_len):
        if not len(func_values) == 2:
            print("Input must have 2 values exactly")
            return

        return (h_len / 2) * (func_values[0] + func_values[1])

    @staticmethod
    def integral_simpson_1_3(func_values, h_len=1):
        if not len(func_values) == 3:
            print("Input must have 3 values exactly")
            return

        return (h_len / 3) * (func_values[0] + 4 * func_values[1] + func_values[2])

    @staticmethod
    def integral_simpson_3_8(func_values, h_len=1):
        if not len(func_values) == 4:
            print("Input must have 4 values exactly")
            return

        return (h_len * 3 / 8) * (func_values[0] + 3 * func_values[1] + 3 * func_values[2] + func_values[3])


def calc_power_mean(series, p_mean_ord=1):
    return np.linalg.norm(series, ord=p_mean_ord) * len(series)**(-1/p_mean_ord)


def return_minkowski_dist_func(norm_order):
    return gen_cust_dist_func(
        lambda a, b: (a-b)**norm_order,
        lambda acc: acc**(1/norm_order),
        parallel=True
    )


def gen_cust_dist_func(kernel_inner, kernel_outer, parallel=True):
    kernel_inner_nb = njit(kernel_inner, fastmath=True, inline='always')
    kernel_outer_nb = njit(kernel_outer, fastmath=True, inline='always')

    def cust_dot_t(A, B):
        assert B.shape[1] == A.shape[1]

        out = np.empty((A.shape[0], B.shape[0]), dtype=A.dtype)
        for i in prange(A.shape[0]):
            for j in range(B.shape[0]):
                acc = 0
                for k in range(A.shape[1]):
                    acc += kernel_inner_nb(A[i, k], B[j, k])
                out[i, j] = kernel_outer_nb(acc)
        return out

    return njit(cust_dot_t, fastmath=True, parallel=parallel)


def gen_cust_p_dist_func(kernel_inner, kernel_outer, parallel=True):
    kernel_inner_nb = njit(kernel_inner, fastmath=True, inline='always')
    kernel_outer_nb = njit(kernel_outer, fastmath=True, inline='always')

    def cust_dot_t(matrix):
        n = matrix.shape[0]
        total_dist_pers = (n * (n - 1)) // 2
        dists_array = np.empty(total_dist_pers, dtype=matrix.dtype)
        counter = 0
        for i in prange(n):
            for j in range(i+1, n):
                acc = 0
                for k in range(matrix.shape[1]):
                    acc += kernel_inner_nb(matrix[i, k], matrix[j, k])
                dists_array[counter] = kernel_outer_nb(acc)
                counter += 1

        return dists_array

    return njit(cust_dot_t, fastmath=True, parallel=parallel)


def linspace_list(start, stop, num, endpoint=True, retstep=False):
    # the same as np.linspace but for lists

    if not endpoint:
        num -= 1

    step = (stop - start) / (num - 1)
    res = [start + i * step for i in range(num)]

    if retstep:
        return res, step

    return res


@njit(fastmath=True, parallel=True)
def trapezoid_area_fn_diff(y_1, y_2, dx=1):
    area_dif = 0
    for x in prange(len(y_1) - 1):
        ac_len = y_1[x] - y_2[x]
        db_len = y_1[x + 1] - y_2[x + 1]
        lb1 = 1 if ac_len > 0 else -1
        lb2 = 1 if db_len > 0 else -1
        ac_len = abs(ac_len)
        db_len = abs(db_len)
        if lb1 * lb2 > 0:
            area_dif += 0.5 * (ac_len + db_len) * dx
        else:
            ad_len = abs(y_1[x] - y_1[x + 1])
            bc_len = abs(y_2[x] - y_2[x + 1])
            xp = ac_len / (ad_len + bc_len)
            # area_dif += 0.5 * abs(ac_len) * xp * dx + 0.5 * (1 - xp) * abs(db_len) * dx
            area_dif += 0.5 * (ac_len * xp + db_len - xp * db_len) * dx

    return area_dif


def geometric_median(X, eps=1e-5):
    """
    Yehuda Vardi and Cun-Hui Zhang's algorithm for the geometric median,
    described in the paper "The multivariate L1-median and associated data depth"
    https://www.pnas.org/doi/pdf/10.1073/pnas.97.4.1423
    """

    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
