import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Optional, Callable, Literal
import re
import time


@dataclass
class MetricsTree:
    metric_name: str
    percent: float
    sub_obj_list: Optional[list["MetricsTree"]] = None

    def update_percent(self, metric_name, percent):
        if self.metric_name == metric_name:
            self.percent = percent
            return True

        for sub_obj in self.sub_obj_list:
            if sub_obj.update_percent(metric_name, percent):
                return True

        return False

    def add_branch(self, metric_tree_obj, father_metric_name):
        if self.metric_name == father_metric_name:
            self.sub_obj_list += [metric_tree_obj]
            return True

        for sub_obj in self.sub_obj_list:
            if sub_obj.add_branch(metric_tree_obj, father_metric_name):
                return True

        return False

    def weights(self, metrics_list, non_zero=True):
        if self.sub_obj_list is None:
            weights_list = []
            m_names = []
            if self.metric_name in metrics_list:
                weights_list = [self.percent]
                m_names = [self.metric_name]
            elif non_zero:
                weights_list = [self.percent] if self.metric_name in metrics_list else [0.0]
                m_names = [self.metric_name]

            return m_names, weights_list

        weights_list = []
        m_names = []
        for sub_obj in self.sub_obj_list:
            m_n, m_p = sub_obj.weights(metrics_list)
            m_names += m_n
            weights_list += m_p

        sum_of_w = np.sum(weights_list)
        if sum_of_w == 0:
            return m_names, weights_list

        weights_list = list(np.array(weights_list) * (self.percent / sum_of_w))
        return m_names, weights_list


def print_execution_time(func):
    def wrapper(*args, **kwargs):
        time_st = time.time()
        perf_st = time.perf_counter()
        exec_st = time.process_time()
        res = func(*args, **kwargs)
        time_end = time.time()
        perf_end = time.perf_counter()
        exec_end = time.process_time()
        print(
            f"""
            Total time taken in fn: {func.__name__} | time: {time_end - time_st}
            performance time: {perf_end - perf_st}
            process_time: {exec_end - exec_st}\n
            """
        )
        return res

    return wrapper


def type_check(class_instance, class_to_check):
    return type(class_instance).__name__ == class_to_check.__name__


def add_name_to_dataframe_num_index(dataframe, name, inplace=True):
    new_index = []
    for i in dataframe.index:
        if isinstance(i, int):
            new_index += ["{}_{}".format(name, i)]

    if inplace is True:
        dataframe.set_index([new_index], inplace=True)
    elif inplace is False:
        new_dataframe = dataframe.set_index([new_index], inplace=False)
        return new_dataframe
    else:
        print("Variable 'inplace' value should be 'True' or 'False'")


def all_lists_elements_combinations(*lists):
    len_lists = len(lists)
    new_list = []
    # if len_lists == 1:
        # return lists

    if len_lists == 2:
        for l1 in lists[0]:
            for l2 in lists[1]:
                if not isinstance(l1, list):
                    if not isinstance(l2, list):
                        new_list += [[l1] + [l2]]
                    else:
                        new_list += [[l1] + l2]
                else:
                    if not isinstance(l2, list):
                        new_list += [l1 + [l2]]
                    else:
                        new_list += [l1 + l2]
        return new_list
    elif len_lists == 3:
        return all_lists_elements_combinations(
            lists[0],
            all_lists_elements_combinations(lists[1], lists[2])
        )
    else:
        return all_lists_elements_combinations(
            all_lists_elements_combinations(*lists[int(len_lists / 2):]),
            all_lists_elements_combinations(*lists[:int(len_lists / 2)])
        )


def label_encoder(labels: pd.Series | list, unique_vals: pd.Index | pd.MultiIndex | np.ndarray | list | tuple):
    labels_c = []
    for row in labels:
        for j, col in enumerate(unique_vals):
            if row == col:
                labels_c += [j]
    return labels_c


def def_var_value_if_none(value_passed, default=None, def_func=None):
    if value_passed is None:
        if default is not None:
            return default
        elif callable(def_func):
            return def_func()
        else:
            raise Exception("'default' value is None and 'def_func' not callable")
    else:
        return value_passed


def extract_series_from_obj_dict(obj_dict: dict, var_name: str, axis=0, convert_func=None) -> pd.DataFrame:
    if convert_func is None:
        convert_func = lambda x: x

    if axis == 0:
        return pd.DataFrame([convert_func(getattr(obj, var_name)).rename(key) for key, obj in obj_dict.items()])
    elif axis == 1:
        return pd.concat(
            [
                convert_func(getattr(obj, var_name)).rename(key)
                for key, obj in obj_dict.items()
            ],
            keys=obj_dict.keys(),
            axis=1
        )
    else:
        """
        print("line_1028_mf")
        for key, obj in obj_dict.items():
            print("\n--\n", obj.__dict__[var_name], "\n--\n")
            print("\n--\n", convert_func(obj.__dict__[var_name]), "\n--\n")
        """
        return pd.concat(
            [
                convert_func(getattr(obj, var_name)).rename(key)
                for key, obj in obj_dict.items()
            ],
            keys=obj_dict.keys(),
            axis=0
        )


def filter_dict_by_keys(my_dict, keys_list):
    return {key: my_dict[key] for key in keys_list}


def get_dict_0key_val(dict_of_objs):
    return dict_of_objs[list(dict_of_objs.keys())[0]]


def n_min(array, n):
    portion = array[:n]
    max_poss = np.argmax(portion)
    poss = np.array(range(n))
    for i in range(len(array) - n):
        if array[n + i] < portion[max_poss]:
            portion[max_poss] = array[n + i]
            poss[max_poss] = n + i
            max_poss = np.argmax(portion)

    return portion, poss


def sort_string_list_based_on_number_inside(string_list: list):
    """
    if return_int_number_of_string(string_list[0]) is None:
        list_keys_list = string_list
    else:
        list_keys_list = sorted(string_list,
                                key=return_int_number_of_string)  # lambda a: return_int_number_of_string(a))
    """

    if not isinstance(string_list, list):
        return

    if not string_list:
        return

    if return_int_number_of_string(string_list[0]) is None:
        return string_list

    list_keys_list = sorted(string_list, key=return_int_number_of_string)
    return list_keys_list


def print_dictionary(my_dict):
    if not isinstance(my_dict, dict):
        print(my_dict)
        return
    if my_dict == {}:
        print("Empty dictionary")
        return

    list_keys_list = sort_string_list_based_on_number_inside(list(my_dict.keys()))

    print("\n{{")
    for i in list_keys_list:
        print("{"+f"\n{i} : ")
        if isinstance(my_dict[i], list):
            print_list_matrix(my_dict[i])
        elif isinstance(my_dict[i], dict):
            print_dictionary(my_dict[i])
        else:
            print(my_dict[i])
        print("}")
    print("\n}}\n")


def print_list_matrix(list_matrix):
    if not isinstance(list_matrix, list):
        print("Value pasted is not a list:", list_matrix)
        return

    if not isinstance(list_matrix[0], list):
        print(list_matrix)
        return

    str_list = []
    for i in list_matrix:
        rows = []
        if not isinstance(i, list):
            print(list_matrix)
            return
        else:
            for j in i:
                if isinstance(j, (int, float)):
                    rows += [float("{:.2f}".format(j))]
                else:
                    rows += [j]
            str_list += [rows]

    print("list_matrix:")
    for k in str_list:
        print(k)


def print_list(my_list):
    if not isinstance(my_list, list):
        print("Input is not a list, type of input: {}".format(type(my_list)))
    if not my_list:
        print("List is empty", my_list)

    is_list_matrix = True
    for i in my_list:
        if not isinstance(i, list):
            is_list_matrix = False
    if is_list_matrix:
        print_list_matrix(my_list)
        return

    for i in my_list:
        if is_list_matrix:
            print(i)
        if isinstance(i, list):
            print_list(i)
        elif isinstance(i, dict):
            print_dictionary(i)
        else:
            print(i)
    return


def random_choice_2d(matrix, size: int, replace=False):
    if size > len(matrix):
        return matrix

    return [
        matrix[i] for i in np.random.choice(
            len(matrix),
            size=size,
            replace=replace
        )
    ]


def return_int_number_of_string(string, all_numbers=False, negatives=False):
    if not isinstance(string, str):
        print("Value passed is not a string:")
        print(string)
        print(type(string))
        return string

    num_string = ""
    for i, char in enumerate(string):
        try:
            char_int = str(int(char))
            if i != 0 and negatives is True:
                if string[i - 1] == "-":
                    num_string += "-"
                elif string[i - 1] == "+":
                    num_string += "+"
            num_string += char_int
        except ValueError:
            num_string += "_"

    num = []
    if num_string != "":
        try:
            sting_split = num_string.split("_")
            for i in sting_split:
                if i == "":
                    pass
                else:
                    num += [int(i)]
        except ValueError:
            print("num_string not a string of numbers")

    if not num:
        return None
    if all_numbers is False:
        return num[0]
    elif all_numbers is True:
        return num
    else:
        print("all_nums must be True of False")
        return num


def return_min_num_of_n_power_combs(n_sets, min_comb_set_size):
    depth_n_cp_pairs_slt_n_cps = []
    for i in range(n_sets + 2, 2, -1):
        depth = return_min_num_of_pow_greater_than_min_val(power=i, num_max=20, min_val=min_comb_set_size)
        if depth is not None:
            depth_n_cp_pairs_slt_n_cps += [(depth, i)]
        else:
            break
    return depth_n_cp_pairs_slt_n_cps


def return_min_num_of_pow_greater_than_min_val(power, min_val, num_max=100):
    if power < 2:
        return min_val + 1
    for i in range(2, num_max):
        if i**power > min_val:
            return i
    print(f"max pow of num < min_val: {num_max-1}^{power} == {(num_max-1)**power} < {min_val}")


def return_min_pow_of_num_greater_than_min_val(num, min_val, pow_max):
    for i in range(1, pow_max):
        if num**i > min_val:
            return i
    print(f"max pow of num < min_val: {num}^{pow_max-1} == {num**pow_max-1} < {min_val}")


def return_min_pow_of_n_num_combs(n_sets, min_comb_set_size):
    return [
        (i, return_min_pow_of_num_greater_than_min_val(num=i, pow_max=10, min_val=min_comb_set_size))
        for i in range(2, n_sets+2)
    ]


def rtn_dict_update(dict1: dict, dict2: dict, copy=True):
    if copy:
        m_dict = dict1.copy()
    else:
        m_dict = dict1

    m_dict.update(dict2)

    return m_dict


def select_n_random_rows_from_df(dframe, n_row: int):
    return dframe.loc[list(
        np.random.choice(dframe.index, size=n_row, replace=False)
    )]


def set_up_df_to_index(
        df: pd.DataFrame | pd.Series, idx: pd.Index | pd.MultiIndex, idx_lvl_1=None, agr_apply_f=None
):
    m_idx_df = isinstance(df.index, pd.MultiIndex)
    m_idx_w = isinstance(idx, pd.MultiIndex)

    if m_idx_df == m_idx_w:
        return df.loc[idx]  # .sort_index()

    idx_lvl_1 = def_var_value_if_none(idx_lvl_1, def_func=lambda: list(idx.get_level_values(0).unique()))

    if m_idx_df and not m_idx_w:
        agr_apply_f = def_var_value_if_none(agr_apply_f, default=np.mean)
        return df.loc[idx_lvl_1].groupby(level=0).apply(agr_apply_f)    # .sort_index()

    if not m_idx_df and m_idx_w:
        return df.loc[idx_lvl_1]    # .sort_index()


def sort_str_num_index(df: pd.DataFrame | pd.Series, axis=1, ascending=True):
    return df.sort_index(
            axis=axis,
            key=lambda x: x.map(lambda y: int(re.findall("[0-9]+", y)[0])),     # x.map(mf.return_int_number_of_string)
            ascending=ascending
        )


def tuple_to_text(my_tuple, sep=" - "):
    text = f"{my_tuple[0]}"
    for y in my_tuple[1:]:
        text += f"{sep}{y}"
    return text


def unique_list_vals_filter(flt_func, list_of_vals):
    return list(filter(flt_func, np.unique(list_of_vals)))


def get_last_name_of(obj: str | Callable):
    if isinstance(obj, str):
        full_name = obj
    else:
        full_name = getattr(obj, "__name__", False)
        if not full_name:
            full_name = getattr(obj, "name")

    return full_name[last_period_pos(full_name)+1:]


def last_period_pos(string):
    for i in range(len(string)-1, 0, -1):
        if string[i] == ".":
            return i


def list_of_names_in_list(names_list: list, search_list):
    name = None
    for sn in names_list:
        if sn in search_list:
            name = sn
            break
    return name