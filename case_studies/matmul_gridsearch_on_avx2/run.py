import accera as acc
import hatlib as hat

import os
import shutil
import argparse

from utils import fits_in_l2, uses_enough_l2, valid_split_size
from utils import get_optimal_parameters

def add_matmul_functions(M, N, S, package, parameter_choices, filter_func=None, sample=0):
    '''
        This function is an Accera implementation of multiplying two matrices A and B, A has shape M*S and B has shape S*N.
        Python implementation:
        ---------------------
        for i in range(M):
            for j in range(N):
                for k in range(S):
                    C[i,j] = A[i,k] * B[k,j]
    '''
    A = acc.Array(role=acc.Array.Role.INPUT,element_type=acc.ScalarType.float32, shape=(M, S))
    B = acc.Array(role=acc.Array.Role.INPUT,element_type=acc.ScalarType.float32, shape=(S, N))
    C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT,element_type=acc.ScalarType.float32, shape=(M, N))
    
    p_m_split_size, p_n_split_size, p_s_split_size, \
        p_s_split_2_size, p_n_split_2_size, p_n_split_3_size = acc.create_parameters(6)

    # Create parameters list
    parameters_list = acc.create_parameter_grid({
        p_m_split_size: parameter_choices[0],
        p_n_split_size: parameter_choices[1], 
        p_s_split_size: parameter_choices[2],
        p_s_split_2_size: parameter_choices[3],
        p_n_split_2_size: parameter_choices[4],
        p_n_split_3_size: parameter_choices[5]
    }, filter_func, sample)

    # Define a simple affine loop nest and name its loops i, j, k
    nest = acc.Nest(shape=(M, N, S))
    i, j, k = nest.get_indices()

    # Define the logic of each iteration in the nest
    @nest.iteration_logic
    def _():
        C[i, j] += A[i, k] * B[k, j]

    schedule = nest.create_schedule()

    # Tile splits to place some blocks of the input and output matrices in the L2 cache
    ii, jj, kk = schedule.tile({i: p_m_split_size, j: p_n_split_size, k: p_s_split_size})
    
    # Kernel splits
    kkk = schedule.split(kk, p_s_split_2_size)
    jjj = schedule.split(jj, p_n_split_2_size)
    jjjj = schedule.split(jjj, p_n_split_3_size)

    schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)

    plan = schedule.create_plan()

    plan.cache(B, index=jj)
    plan.cache(C, index=ii)

    # Unroll the non-vectorized kernel loops
    plan.unroll(ii)
    plan.unroll(jjj)

    # Vectorize the innermost kernel loop
    plan.vectorize(jjjj)
    
    function = package.add(plan, args=(A, B, C), parameters=parameters_list, base_name=f"matmul_{M}_{N}_{S}")

    return function


def filter_function(parameters_choice):
    l2_cache_size = 256
    element_size = 4
    m_split_size, n_split_size, s_split_size, s_split_2_size, n_split_2_size, n_split_3_size = parameters_choice

    return valid_split_size(n_split_size, n_split_2_size) \
        and valid_split_size(n_split_2_size, n_split_3_size) \
        and valid_split_size(s_split_size, s_split_2_size) \
        and fits_in_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size) \
        and uses_enough_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Accera Matrix Multiplication Grid Search Case Study')
    parser.add_argument('--matmul_dim', dest='matmul_dim', type=int, nargs='+', help='Dimensions of the MatMul.')
    parser.add_argument('--output_directory', type=str, default="matmul_pkg", help='Output directory.')
    parser.add_argument('--sample', type=int, default=None, help='Optional parameter to choose a number of sample points of the parameter grid.')
    args = parser.parse_args()

    if os.path.isdir(args.output_directory):
        shutil.rmtree(args.output_directory)
    M, N, S = args.matmul_dim

    # Create a accera package using the parameters grid
    package = acc.Package()  
    add_matmul_functions(M=M, N=N, S=S, package=package, 
        parameter_choices=[
        [4, 6, 8, 16, 32, 64, 128, 256], # p_m_split_size
        [4, 6, 8, 16, 32, 64, 128, 256], # p_n_split_size
        [4, 6, 8, 16, 32, 64, 128, 256], # p_s_split_size
        [4, 8, 16], # p_s_split_2_size
        [4, 8, 16], # p_n_split_2_size
        [4, 8, 16]  # p_n_split_3_size
    ], filter_func=filter_function, sample=args.sample)
    package.build("matmul", format=acc.Package.Format.HAT_DYNAMIC, output_dir=args.output_directory)

    # Run benchmark
    hat_file_path = os.path.join(args.output_directory, "matmul.hat")
    hat.run_benchmark(hat_file_path, store_in_hat=True, batch_size=5)

    # Create a new Accera package using the optimal parameters
    optimal_package = acc.Package()
    add_matmul_functions(M=M, N=N, S=S, package=optimal_package, parameter_choices=get_optimal_parameters(hat_file_path))
    optimal_package.build("matmul", format=acc.Package.Format.HAT_DYNAMIC, output_dir=args.output_directory + "_optimal")