from hatlib import HATPackage
import numpy as np

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

            data_point["p_out_c_split_size"] = int(function.auxiliary['accera']['parameters']["p_out_c_split_size"])
            data_point["p_out_r_split_size"] = int(function.auxiliary['accera']['parameters']["p_out_r_split_size"])
            data_point["p_out_f_split_size"] = int(function.auxiliary['accera']['parameters']["p_out_f_split_size"])

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
    return [[int(optimal_point["p_out_c_split_size"])],
            [int(optimal_point["p_out_r_split_size"])],
            [int(optimal_point["p_out_f_split_size"])]]