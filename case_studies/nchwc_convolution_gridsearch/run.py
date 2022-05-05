import accera as acc
import hatlib as hat

import os
import math
import shutil
import argparse

from utils import get_optimal_parameters


def add_nchwc_conv2d_function(nchwc_input_shape, nchwc_output_shape, weights_shape, row_stride, column_stride, package, parameters_choices, filter_func=None, sample=0):
    '''
        This function is a accera implementation for nchwc 2D convolution between
        a 4D input Tensor of dimensions input_channels_blocks*input_rows*input_columns*block_input_channels
        and a 4D Weights Tensor of dimensions kernel_rows*kernel_columns*input_channels*output_filters
        resulting in a 4D output Tensor of dimensions output_filters_blocks*output_rows*output_columns*block_output_filters.
        The logic for the NCHWc 2D covolution can be expressed in python as
        ---------------------
        Python implemntation:
        ---------------------
        for out_f in range(output_filters_blocks):
            for in_ch in range(input_channels_blocks):
                for out_r in range(output_rows):
                    for out_c in range(output_columns):
                        for out_f_b in range(block_output_filters):
                            for in_ch_b in range(block_input_channels):
                                for k_r in range(kernel_rows):
                                    for k_c in range(kernel_columns):
                                        in_r = out_r * row_stride + k_r
                                        in_c = out_c * column_stride + k_c
                                        if in_r >= 0 and in_r < input_rows and in_c >= 0 and in_c < input_columns:
                                            Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] * \
                                                                                    Weights[k_r, k_c, in_ch * input_block_channels + in_ch_b,
                                                                                                out_f * output_block_filters + out_f_b]
    '''
    input_channel_blocks, input_rows, input_columns, input_channels = nchwc_input_shape
    nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block = nchwc_output_shape
    kernel_rows, kernel_columns, total_input_channels, total_output_filters = weights_shape

    Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                shape=(input_channel_blocks, input_rows, input_columns, input_channels))
    Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
                shape=(kernel_rows, kernel_columns, total_input_channels, total_output_filters))
    Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, \
                shape=(nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block))
    
    # TODO: Include the loop order as a parameter to search for
    p_out_c_split_size, p_out_r_split_size, p_out_f_split_size = acc.create_parameters(3)

    # Create parameters list
    parameters_list = acc.create_parameter_grid({
        p_out_c_split_size: parameters_choices[0],
        p_out_r_split_size: parameters_choices[1],
        p_out_f_split_size: parameters_choices[2]
    }, filter_func, sample)

    # Define 8 nested loops for the 8 dimensions
    nest = acc.Nest(shape=(nchwc_output_filters, input_channel_blocks, output_rows, kernel_rows, kernel_columns, \
                          input_channels, output_columns, nchwc_output_filters_block))

    out_f, in_ch, out_r, k_r, k_c, in_ch_b, out_c, out_f_b = nest.get_indices()

    # Define the logic of each iteration in the nest
    @nest.iteration_logic
    def _():
        in_r = out_r * row_stride + k_r
        in_c = out_c * column_stride + k_c
        Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] * \
                                Weights[k_r, k_c, in_ch * input_channels + in_ch_b,
                                        out_f * nchwc_output_filters_block + out_f_b]

    # Create the schedule
    schedule = nest.create_schedule()

    out_c2 = schedule.split(out_c, p_out_c_split_size)
    out_r2 = schedule.split(out_r, p_out_r_split_size)
    out_f2 = schedule.split(out_f, p_out_f_split_size)

    schedule.reorder(out_f, in_ch, out_r, out_r2, out_c, k_r, k_c, in_ch_b, out_f2, out_c2, out_f_b) # apply re-ordering

    plan = schedule.create_plan()

    plan.cache(Input, index=in_ch_b)
    plan.cache(Output, index=out_f2)
    plan.cache(Weights, index=in_ch_b)

    plan.kernelize(unroll_indices=(in_ch_b, out_f2, out_c2), vectorize_indices=out_f_b)

    name = f"conv2d_{input_rows}_{input_columns}_{total_input_channels}_{total_output_filters}"
    package.add(plan, args=(Input, Weights, Output), parameters=parameters_list, base_name=name)


def main():
    parser = argparse.ArgumentParser(description='Accera NCHWc 2D Convolution Case Study')
    parser.add_argument('--input_shape', dest='input_shape', type=int, nargs='+',
                        help='Dimensions of the input 3D Matrix for Convolution.')
    parser.add_argument('--kernel_shape', dest='kernel_shape', type=int, nargs='+',
                        help='Dimensions of the 2D kernel for Convolution.')
    parser.add_argument('--output_filters', type=int,
                        help='Number of the required output filters from Convolution.')
    parser.add_argument('--stride', type=int, nargs='+',
                        help='Row and Column stride for Convolution.')
    parser.add_argument('--output_directory', type=str, default="nchwc_2d_conv",
                        help='Output directory.')
    parser.add_argument('--sample', type=int, default=None,
                        help='Optional parameter to choose a number of sample points of the parameter grid.')
    args = parser.parse_args()

    input_rows, input_columns, total_input_channels = args.input_shape
    kernel_rows, kernel_columns = args.kernel_shape
    row_stride, column_stride = args.stride

    total_output_filters = args.output_filters
    output_rows = int((input_rows - kernel_rows)/row_stride + 1)
    output_columns = int((input_columns - kernel_columns)/column_stride + 1)

    # For this case study we choose the NCHWc input channels block size,
    # as well as NCHWc output filters block size to be 8
    # because they optimize the usage of SIMD instructions in the target architecture (AVX2)
    # but for different architectures like (AVX512), 16 would be a better choice.
    output_filters_block_size = 8
    input_channels_block_size = 8

    output_filters_blocks = math.ceil(total_output_filters/output_filters_block_size)
    input_channels_blocks = math.ceil(total_input_channels/input_channels_block_size)

    nchwc_input_shape = [input_channels_blocks, input_rows, input_columns, input_channels_block_size]
    nchwc_output_shape = [output_filters_blocks, output_rows, output_columns, output_filters_block_size]
    weights_shape = [kernel_rows, kernel_columns, total_input_channels, total_output_filters]

    if os.path.isdir(args.output_directory):
        shutil.rmtree(args.output_directory)

    # Create a accera package using the parameters grid
    package = acc.Package()  
    add_nchwc_conv2d_function(nchwc_input_shape=[input_channels_blocks, input_rows, input_columns, input_channels_block_size], \
                              nchwc_output_shape=[output_filters_blocks, output_rows, output_columns, output_filters_block_size], \
                              weights_shape=[kernel_rows, kernel_columns, total_input_channels, total_output_filters], \
                              row_stride=row_stride, column_stride=column_stride, \
                              package=package, \
                              parameters_choices=[
                                  [1, 3, 5, 7], # p_out_c_split_size
                                  [1, 3, 5, 7], # p_out_r_split_size
                                  [4, 8, 16, 32] # p_out_f_split_size
                              ], sample=args.sample)
    package.build("nchwc", format=acc.Package.Format.HAT_DYNAMIC, output_dir=args.output_directory)

    # Run benchmark
    hat_file_path = os.path.join(args.output_directory, "nchwc.hat")
    hat.run_benchmark(hat_file_path, store_in_hat=True, batch_size=5)

    # Create a new Accera package using the optimal parameters
    optimal_package = acc.Package()
    add_nchwc_conv2d_function(nchwc_input_shape=[input_channels_blocks, input_rows, input_columns, input_channels_block_size], \
                              nchwc_output_shape=[output_filters_blocks, output_rows, output_columns, output_filters_block_size], \
                              weights_shape=[kernel_rows, kernel_columns, total_input_channels, total_output_filters], \
                              row_stride=row_stride, column_stride=column_stride, \
                              package=optimal_package, \
                              parameters_choices=get_optimal_parameters(hat_file_path))

    optimal_package.build("nchwc", format=acc.Package.Format.HAT_DYNAMIC, output_dir=args.output_directory + "_optimal")

if __name__ == "__main__":
    main()