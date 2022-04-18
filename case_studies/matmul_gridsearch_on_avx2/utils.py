from hatlib import HATPackage
import numpy as np


def valid_split_size(first_split_size, second_split_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the second split size is smaller than the first split size.
    '''
    return second_split_size < first_split_size

def fits_in_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the total active tiles memory is smaller than the L2 cache size.
        element_size: element size in bytes (i.e. 4 if we assume float32 elements)
        l2_cache_size: cache size in KB
    '''
    tile_mem = element_size*(m_split_size*s_split_size + s_split_size*n_split_size + m_split_size*n_split_size) / 1024
    return tile_mem < l2_cache_size

def uses_enough_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size):
    '''
        This function is one of the parameters grid filtering utils.
        It returns True if the total active tiles memory is at least 50% of the L2 cache size.
        element_size: element size in bytes (i.e. 4 if we assume float32 elements)
        l2_cache_size: cache size in KB
    '''
    tile_mem = element_size*(m_split_size*s_split_size + s_split_size*n_split_size + m_split_size*n_split_size) / 1024
    return tile_mem >= 0.5 * l2_cache_size

def get_auxiliary_data(directory):
    '''
        return a list of the functions auxilary data in a Accera package
    '''
    hat_package = HATPackage(directory)
    functions = [fn for fn in hat_package.get_functions()]

    data = []
    for function in functions:
        if "mean_duration_in_sec" in function.auxiliary: # Only add if the function has benchmarking results
            data_point={}
            data_point["mean_duration_in_sec"] = function.auxiliary["mean_duration_in_sec"]

            data_point["p_m_split_size"] = int(function.auxiliary['accera']['parameters']["p_m_split_size"])
            data_point["p_n_split_size"] = int(function.auxiliary['accera']['parameters']["p_n_split_size"])
            data_point["p_s_split_size"] = int(function.auxiliary['accera']['parameters']["p_s_split_size"])
            data_point["p_n_split_2_size"] = int(function.auxiliary['accera']['parameters']["p_n_split_2_size"])
            data_point["p_n_split_3_size"] = int(function.auxiliary['accera']['parameters']["p_n_split_3_size"])
            data_point["p_s_split_2_size"] = int(function.auxiliary['accera']['parameters']["p_s_split_2_size"])

            data_point["tile_volume"] = 4*(int(data_point["p_m_split_size"])*int(data_point["p_n_split_size"]) + \
                                           int(data_point["p_m_split_size"])*int(data_point["p_s_split_size"]) + \
                                           int(data_point["p_n_split_size"])*int(data_point["p_s_split_size"]))/1024
            
            data.append(data_point)
    return data

def load_to_dataframe(data):
    import pandas as pd
    df = pd.DataFrame(data=data)
    indices = np.arange(0, len(df))
    df["idx"] = indices
    return df

def get_optimal_parameters(hat_file_path):
    data = get_auxiliary_data(hat_file_path)
    if not data:
        print(f"WARNING: No Benchmarking results found in {hat_file_path}")
        return
    dataframe = load_to_dataframe(data)
    optimal_point_idx = dataframe['mean_duration_in_sec'].idxmin()
    optimal_point = dataframe.iloc[optimal_point_idx]
    print("Optimal point by Grid Search:")
    print(optimal_point)
    return [[int(optimal_point["p_m_split_size"])],
            [int(optimal_point["p_n_split_size"])],
            [int(optimal_point["p_s_split_size"])],
            [int(optimal_point["p_s_split_2_size"])],
            [int(optimal_point["p_n_split_2_size"])],
            [int(optimal_point["p_n_split_3_size"])]]