import accera as acc
import hatlib as hat

import os
import shutil
import argparse

from utils import fits_in_l2, uses_enough_l2, valid_split_size
from utils import get_optimal_parameters


def add_unrolled_conv2d_function(input_shape, kernel_shape, output_filters, row_stride, column_stride, package, parameters_choices, filter_func=None, sample=0):
    '''
    # The logic for the 2D covolution can be expressed in python as follows
    for out_f in range(output_filters):
        for out_r in range(output_rows):
            for out_c in range(output_columns):
                for in_ch in range(input_channels):
                    for k_r in range(kernel_rows):
                        for k_c in range(kernel_columns):
                            in_r = out_r * row_stride + k_r
                            in_c = out_c * column_stride + k_c
                            if in_r >= 0 and in_r < input_rows and in_c >= 0 and in_c < input_columns:
                                Output[out_r, out_c, out_f] += Input[in_r, in_c, in_ch] * Weights[k_r, k_c, in_ch, out_f]
    '''

    input_rows, input_columns, input_channels = input_shape
    kernel_rows, kernel_columns = kernel_shape

    output_rows = int(((input_rows - kernel_rows) / row_stride) + 1)
    output_columns = int(((input_columns - kernel_columns) / column_stride) + 1)

    Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32,
                      shape=(input_rows, input_columns, input_channels))
    Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32,
                        shape=(kernel_rows, kernel_columns, input_channels, output_filters))
    Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32,
                       shape=(output_rows, output_columns, output_filters))

    p_outf_split1_size, p_outf_split2_size, p_outf_split3_size, \
                p_outc_split_size, p_in_ch_split_size = acc.create_parameters()

    # Create parameters list
    parameters_list = acc.create_parameter_grid({
        p_outf_split1_size: parameters_choices[0],
        p_outf_split2_size: parameters_choices[1],
        p_outf_split3_size: parameters_choices[2],
        p_outc_split_size: parameters_choices[3],
        p_in_ch_split_size: parameters_choices[4]
    }, filter_func, sample)

    # Define 6 nested loops for the 6 dimensions
    nest = acc.Nest(shape=(output_filters, output_rows, output_columns,
                           input_channels, kernel_rows, kernel_columns))

    out_f, out_r, out_c, in_ch, k_r, k_c = nest.get_indices()

    # Define the logic of each iteration in the nest
    @nest.iteration_logic
    def _():
        in_r = out_r * row_stride + k_r
        in_c = out_c * column_stride + k_c
        Output[out_r, out_c, out_f] += Input[in_r, in_c, in_ch] * Weights[k_r, k_c, in_ch, out_f]

    # Create the schedule
    schedule = nest.create_schedule()

    out_f2 = schedule.split(out_f, p_outf_split1_size)  # jj
    out_f3 = schedule.split(out_f2, p_outf_split2_size)  # jjj
    out_f4 = schedule.split(out_f3, p_outf_split3_size)  # jjjj

    out_c2 = schedule.split(out_c, p_outc_split_size)  # ii
    in_ch2 = schedule.split(in_ch, p_in_ch_split_size)  # kk

    schedule.reorder(out_f,  # j
                     k_r, in_ch,  # k
                     out_r, out_c,  # i
                     out_f2,  # jj
                     in_ch2,  # kk
                     k_c,  # kkk
                     out_c2,  # ii
                     out_f3,  # jjj
                     out_f4  # jjjj
                     )

    plan = schedule.create_plan()

    plan.cache(Input, index=in_ch2)
    plan.cache(Weights, index=out_f2)
    plan.cache(Output, index=out_f2)

    plan.unroll(out_c2)
    plan.unroll(out_f3)
    plan.vectorize(out_f4)

    name = f"conv2d_{input_rows}_{input_columns}_{input_channels}_{output_filters}"
    package.add(plan, args=(Input, Weights, Output), parameters=parameters_list, base_name=name)

def filter_function(parameters_choice):
    l2_cache_size = 256
    element_size = 4
    outf_split1_size, outf_split2_size, outf_split3_size, outc_split_size, in_ch_split_size= parameters_choice

    return valid_split_size(outf_split1_size, outf_split2_size) \
            and valid_split_size(outf_split2_size, outf_split3_size) \
            and fits_in_l2(outc_split_size, outf_split1_size, in_ch_split_size, element_size, l2_cache_size) \
            and uses_enough_l2(outc_split_size, outf_split1_size, in_ch_split_size, element_size, l2_cache_size)

def main():
    parser = argparse.ArgumentParser(
        description='Accera Unrolled 2D Convolution Grid Search Case Study')
    parser.add_argument('--input_shape', dest='input_shape', type=int, nargs='+',
                        help='Dimensions of the input 3D Matrix for Convolution.')
    parser.add_argument('--kernel_shape', dest='kernel_shape', type=int, nargs='+',
                        help='Dimensions of the 2D kernel for Convolution.')
    parser.add_argument('--output_filters', type=int,
                        help='Number of the required output filters from Convolution.')
    parser.add_argument('--stride', type=int, nargs='+',
                        help='Row and Column stride for Convolution.')
    parser.add_argument('--output_directory', type=str, default="unrolled_conv",
                        help='Output directory.')
    parser.add_argument('--sample', type=int, default=10,
                        help='Optional parameter to choose a number of sample points of the parameter grid.')
    args = parser.parse_args()


    input_shape = args.input_shape
    kernel_shape = args.kernel_shape
    output_filters = args.output_filters
    row_stride, column_stride = args.stride

    if os.path.isdir(args.output_directory):
        shutil.rmtree(args.output_directory)

    # Create a accera package using the parameters grid
    package = acc.Package()  

    # Create a accera package using the parameters grid
    package = acc.Package()  
    add_unrolled_conv2d_function(input_shape=input_shape, kernel_shape=kernel_shape, output_filters=output_filters,
                                 row_stride=row_stride, column_stride=column_stride, package=package, \
                                parameters_choices=[
                                    [32, 64, 128, 256], # outf_split1_size
                                    [4, 8, 16, 32], # outf_split2_size
                                    [4, 8, 16, 32], # outf_split3_size
                                    [4, 6, 8], # outc_split_size
                                    [32, 64, 128, 256], # in_ch_split_size
                                ], filter_func=filter_function, sample=args.sample)
    package.build("unrolled_conv", format=acc.Package.Format.HAT_DYNAMIC, output_dir=args.output_directory)

    # Run benchmark
    hat_file_path = os.path.join(args.output_directory, "unrolled_conv.hat")
    hat.run_benchmark(hat_file_path, store_in_hat=True, batch_size=5)

    # Create a new Accera package using the optimal parameters
    optimal_package = acc.Package()
    add_unrolled_conv2d_function(input_shape=input_shape, kernel_shape=kernel_shape, output_filters=output_filters,
                                 row_stride=row_stride, column_stride=column_stride, package=optimal_package, \
                                 parameters_choices=get_optimal_parameters(hat_file_path))

    optimal_package.build("unrolled_conv", format=acc.Package.Format.HAT_DYNAMIC, output_dir=args.output_directory + "_optimal")

if __name__ == "__main__":
    main()