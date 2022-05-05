[//]: # (Project: Accera)
[//]: # (Version: 1.2.3)

# Case Study - NCHWc 2D Convolution Grid Search

In this case study, we will discuss how to construct a performant implementation for NCHWc 2D Convolution on an AVX2 Machine using [Accera](https://microsoft.github.io/Accera/). First, we will show how to create a parameterized Accera schedule and plan. Then we will discuss how to create a parameters grid, and how to benchmark each point in the grid in order to pick the best performant implementation.

A 2D convolution between a 3D input tensor of dimensions `input_rows`&times;`input_columns`&times;`input_channels` and a 4D weights tensor of dimensions `kernel_rows`&times;`kernel_columns`&times;`input_channels`&times;`output_filters` would result in a 3D output tensor of dimensions `output_rows`&times;`output_columns`&times;`output_filters`.
`output_rows` is defined as `(input_rows - kernel_rows)/row_stride + 1`, while `output_columns` is defined as `(input_columns - kernel_columns)/column_stride + 1`. `row_stride` and `column_stride` control the stride for the convolution. The logic for the 2D covolution can be expressed in python as follows

```python
for out_f in range(output_filters):
    for out_r in range(output_rows):
        for out_c in range(output_columns):
            for in_ch in range(input_channels):
                for k_r in range(kernel_rows):
                    for k_c in range(kernel_columns):
                        in_r = out_r * row_stride + k_r
                        in_c = out_c * column_stride + k_c
                        Output[out_r, out_c, out_f] += Input[in_r, in_c, in_ch] * Weights[k_r, k_c, in_ch, out_f]
```

For NCHWc 2D convolution, the input and output tensors need to be re-ordered from the NHWC format to the NCHWc format. In this case the 2D convolution occurs between a 4D input tensor of dimensions `input_channel_blocks`&times;`input_rows`&times;`input_columns`&times;`input_channels` and a 4D weights tensor of dimensions `kernel_rows`&times;`kernel_columns`&times;`total_input_channels`&times;`total_output_filters` resulting in a 4D output tensor of dimensions `output_filters_blocks`&times;`output_rows`&times;`output_columns`&times;`output_filters`. The logic for the NCHWc 2D covolution can be expressed in python as

```python
for out_f in range(output_filters_blocks):
    for in_ch in range(input_channel_blocks):
        for out_r in range(output_rows):
            for out_c in range(output_columns):
                for out_f_b in range(output_filters):
                    for in_ch_b in range(input_channels):
                        for k_r in range(kernel_rows):
                            for k_c in range(kernel_columns):
                                in_r = out_r * row_stride + k_r
                                in_c = out_c * column_stride + k_c
                                k_ch = in_ch * input_channels + in_ch_b
                                k_f = out_f * output_filters + out_f_b
                                Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] * Weights[k_r, k_c, k_ch, k_f]
```

In this case study, we present the end-to-end steps needed to write a performant implementation for NCHWc 2D Convolution as follows:

- [Step 1 - Create an Accera NCHWc 2D Convolution Function](#step-1---create-an-accera-nchwc-2d-convolution-function)
- [Step 2 - Create a Parameterized Accera NCHWc 2D Convolution Function](#step-2---create-a-parameterized-accera-nchwc-2d-convolution-function)
- [Step 3 - Create a parameters grid](#step-3---create-a-parameters-grid)
- [Step 4 - Create a Accera package with all parameters choices in the parameters grid](#step-4---create-a-accera-package-with-all-parameters-choices-in-the-parameters-grid)
- [Step 5 - Benchmark the package on the target hardware](#step-5---benchmark-the-package-on-the-target-hardware)
- [Step 6 - Find the optimal parameters choice](#step-6---find-the-optimal-parameters-choice)
- [Step 7 - Create a Accera package with the optimal function](#step-7---create-a-accera-package-with-the-optimal-function)
- [Pull it all together](#pull-it-all-together)

## Step 1 - Create an Accera NCHWc 2D Convolution Function
As mentioned earlier, the NCHWc convolution uses a different memory layout for the input and output tensors. The input to the convolution needs to be reordered from the 3D NHWC layout into 4D NCHWc layout by doing a split along the channels dimensions. While the output from the convolution needs to be reordered from the 4D NCHWc tensors back into the initial 3D NHWC memory layout.

Re-ordering the tensor layout from NHWC into NCHWc can be expressed in python as follows:

```python
NCHWc_tensor = np.zeros((nchwc_channels_blocks, rows, columns, nchwc_channels)).astype(np.float32)

for ch in range(nchwc_channels_blocks):
    for r in range(rows):
        for c in range(columns):
            for ch_b in range(nchwc_channels):
                if ch * nchwc_channels + ch_b < total_channels:
                    NCHWc_tensor[ch, r, c, ch_b] = NHWC_tensor[r, c, ch * nchwc_channels + ch_b]
```

Re-ordering the tensor layout from NCHWc back into NHWC can be expressed in python as follows:

```python
NHWC_tensor = np.zeros((rows, columns, total_channels)).astype(np.float32)

for ch in range(nchwc_channels):
    for r in range(rows):
        for c in range(columns):
            for ch_b in range(nchwc_channels_block):
                if ch * nchwc_channels_block + ch_b < total_channels:
                    NHWC_tensor[r, c, ch * nchwc_channels_block + ch_b] = NCHWc_tensor[ch, r, c, ch_b]
```

For this case study, we choose the NCHWc input channels block size as well as NCHWc output filters block size to be 8 because this optimizes the usuage of the SIMD registers in the target architecture (AVX2). However, for different architectures like (AVX512), 16 would be a better choice. We will assume that the input and the output tensors are pre and post re-ordered using external functions, and we will focus on the convolution implementation itself.

Now, we can implement 2D NCHWc convolution using Accera as follows:

```python
import accera as acc

input_channel_blocks, input_rows, input_columns, input_channels = 64, 7, 7, 8
nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block = 64, 5, 5, 8
kernel_rows, kernel_columns, total_input_channels, total_output_filters = 3, 3, 512, 512
row_stride, column_stride = 2, 2

Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
            shape=(input_channel_blocks, input_rows, input_columns, input_channels))
Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
            shape=(kernel_rows, kernel_columns, total_input_channels, total_output_filters))
Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, \
            shape=(nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block))

# Define 8 nested loops for the 8 dimensions
nest = acc.Nest(shape=(nchwc_output_filters, input_channel_blocks, output_rows, \
                       kernel_rows, kernel_columns, \
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

# Define schedule
schedule = nest.create_schedule()

# Define plan
plan = schedule.create_plan()

# Add the function to a package and build the package
name = f"conv2d_{input_rows}_{input_columns}_{total_input_channels}_{total_output_filters}"
package = acc.Package()
package.add(plan, args=(Input, Weights, Output), base_name=name)
package.build(name, format=acc.Package.Format.HAT_DYNAMIC, output_dir=name)
```

This implementation would use the default schedule and plan. This Accera snippet would produce the correct results, but will not produce a high performance implementation. We can improve the efficiency of this Accera implementation using schedule and plan optimizations such as splits, vectorization, unrolling, and caching as we will do in the next step.

## Step 2 - Create a Parameterized Accera NCHWc 2D Convolution Function
For this case study, we choose to do one tiling split per output dimension, then we will also apply caching, unrolling and vectorization to make use of the registers on the target hardware. For this step, we will start with some chosen values for the split sizes, the unrolling factors, and the schedule loop order. However, in the next steps we will discuss how to define a parameter grid and search through it to get the performant implementation for the target hardware.

First, we define the input and output arrays

```python
Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
            shape=(input_channel_blocks, input_rows, input_columns, input_channels))
Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
            shape=(kernel_rows, kernel_columns, total_input_channels, total_output_filters))
Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, \
            shape=(nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block))
```

Then, we define some parameters that would be used later while creating the schedule and the plan.

```python
p_out_c_split_size, p_out_r_split_size, p_out_f_split_size = acc.create_parameters(3)
```

Then we define the iteration logic as:

```python
nest = acc.Nest(shape=(nchwc_output_filters, input_channel_blocks, output_rows, \
                       kernel_rows, kernel_columns, \
                       input_channels, output_columns, nchwc_output_filters_block))
out_f, in_ch, out_r, k_r, k_c, in_ch_b, out_c, out_f_b = nest.get_indices()
@nest.iteration_logic
def _():
    in_r = out_r * row_stride + k_r
    in_c = out_c * column_stride + k_c
    Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] * \
                            Weights[k_r, k_c, in_ch * input_channels + in_ch_b,
                                    out_f * nchwc_output_filters_block + out_f_b]
```

Next, we define the 2D convolution schedule and apply one split per output dimension, and we reorder the schedule's loops.

```python
schedule = nest.create_schedule()

# Add one split along each output dimension
out_c2 = schedule.split(out_c, p_out_c_split_size)
out_r2 = schedule.split(out_r, p_out_r_split_size)
out_f2 = schedule.split(out_f, p_out_f_split_size)

# Apply loop re-ordering
schedule.reorder(out_f, in_ch, out_r, out_r2, out_c, k_r, k_c, in_ch_b, out_f2, out_c2, out_f_b) 
```
Then, we define a plan, and we use caching, unrolling, and vectorization to make the implementation more performant.
For the caching level, convolutional neural networks usually deep layers with large number of filters (except for the first layer). Therefore, it makes more sense to cache along a channels split/block dimension. We choose to cache the Input at the `in_ch_b` dimension, while cache the Output at the `out_f2` dimension. For the Weights, we can choose either `in_ch_b` or `out_f2`.

```python
plan = schedule.create_plan()

plan.cache(Input, index=in_ch_b)
plan.cache(Output, index=out_f2)
plan.cache(Weights, index=in_ch_b)

plan.kernelize(unroll_indices=(in_ch_b, out_f2, out_c2), vectorize_indices=out_f_b)
```

To create a package with this 2D Convolution function, we need to know the values of the parameters `p_out_c_split_size`, `p_out_r_split_size`, `p_out_f_split_size`, and `loop_order`. Let's assume for now that we are given those constants, then we could set those parameters, and add the function to the Accera package and build it. However, we will explain in the next section how to get the right values for them

```python
package = acc.Package()
name = f"conv2d_{input_rows}_{input_columns}_{total_input_channels}_{total_output_filters}"
parameters_values = {p_out_c_split_size:4, p_out_r_split_size:4, p_out_f_split_size:4}
function = package.add(plan, args=(Input, Weights, Output), parameters=parameters_values, base_name=name)
package.build(name, format=acc.Package.Format.HAT_DYNAMIC, output_dir=name)
```

Pulling it all together
```python
import accera as acc

input_channel_blocks, input_rows, input_columns, input_channels = 64, 7, 7, 8
nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block = 64, 5, 5, 8
kernel_rows, kernel_columns, total_input_channels, total_output_filters = 3, 3, 512, 512
row_stride, column_stride = 2, 2

Input = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
            shape=(input_channel_blocks, input_rows, input_columns, input_channels))
Weights = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, \
            shape=(kernel_rows, kernel_columns, total_input_channels, total_output_filters))
Output = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, \
            shape=(nchwc_output_filters, output_rows, output_columns, nchwc_output_filters_block))

p_out_c_split_size, p_out_r_split_size, p_out_f_split_size = acc.create_parameters(3)

nest = acc.Nest(shape=(nchwc_output_filters, input_channel_blocks, output_rows, \
                       kernel_rows, kernel_columns, \
                       input_channels, output_columns, nchwc_output_filters_block))
out_f, in_ch, out_r, k_r, k_c, in_ch_b, out_c, out_f_b = nest.get_indices()
@nest.iteration_logic
def _():
    in_r = out_r * row_stride + k_r
    in_c = out_c * column_stride + k_c
    Output[out_f, out_r, out_c, out_f_b] += Input[in_ch, in_r, in_c, in_ch_b] * \
                            Weights[k_r, k_c, in_ch * input_channels + in_ch_b,
                                    out_f * nchwc_output_filters_block + out_f_b]

schedule = nest.create_schedule()

# Add one split along each output dimension
out_c2 = schedule.split(out_c, p_out_c_split_size)
out_r2 = schedule.split(out_r, p_out_r_split_size)
out_f2 = schedule.split(out_f, p_out_f_split_size)

# Apply loop re-ordering
schedule.reorder(out_f, in_ch, out_r, out_r2, out_c, k_r, k_c, in_ch_b, out_f2, out_c2, out_f_b) 

plan = schedule.create_plan()

plan.cache(Input, index=in_ch_b)
plan.cache(Output, index=out_f2)
plan.cache(Weights, index=in_ch_b)

plan.kernelize(unroll_indices=(in_ch_b, out_f2, out_c2), vectorize_indices=out_f_b)

package = acc.Package()
name = f"conv2d_{input_rows}_{input_columns}_{total_input_channels}_{total_output_filters}"
parameters_values = {p_out_c_split_size:4, p_out_r_split_size:4, p_out_f_split_size:4}
function = package.add(plan, args=(Input, Weights, Output), parameters=parameters_values, base_name=name)
package.build(name, format=acc.Package.Format.HAT_DYNAMIC, output_dir=name)
```

## Step 3 - Create a parameters grid
As mentioned in the previous step, we assumed some values for the split sizes. However, we are not sure that those chosen sizes would give the best performance, also for each different hardware the parameters values that achieves the best performant implementation can be different. To ensure that the created Accera function is performant (i.e. has the right parameters), we define a parameters grid where our chosen parameters are:

1. `p_out_c_split_size`
2. `p_out_r_split_size`
3. `p_out_f_split_size`

and our grid will consist of a set of possible values for those parameters.

For example, we might want to:
1. define the `p_out_c_split_size` and `p_out_r_split_size`as any odd number between 1 and 7 because those dimensions usually have smaller values in convolutional neural networks
2. define the `p_out_f_split_size` as any power of 2 between 4 and 32.

We can create our parameters grid by calling create_parameter_grid method.

```python
parameters_list = acc.create_parameter_grid({
    p_out_c_split_size: [1, 3, 5, 7],
    p_out_r_split_size: [1, 3, 5, 7],
    p_out_f_split_size: [4, 8, 16, 32]   
    })
```

## Step 4 - Create a Accera package with all parameters choices in the parameters grid
In this step, we would create a Accera package with all parameters choices and the filter defined above.

This can be done simply by replacing the below code from [Step 2](#step-2---create-a-parameterized-accera-nchwc-2d-convolution-function)
```python
package.add(plan, args=(Input, Weights, Output), parameters=parameters_values, base_name=name)
```
to 
```python
package.add(plan, args=(Input, Weights, Output), parameters=parameters_list, base_name=name)
```
where `parameters_list` is defined in the previous step.


## Step 5 - Benchmark the package on the target hardware
Finally, we can use [HAT tools](https://github.com/microsoft/hat) to benchmark our Accera package on the target hardware, and write back the timing results to the package.
```python
hat_file_path = os.path.join(package_directory, "nchwc_conv.hat")
hat.run_benchmark(hat_file_path, store_in_hat=write_back, batch_size=5)
```

## Step 6 - Find the optimal parameters choice
In this step, we define a function `get_optimal_parameters` to get the optimal parameters. We achieve this by loading the data from the HAT file to a pandas dataframe, and then the optimal parameters choice would be the one with minimum mean duration in seconds.
```python
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
```
> **_Note_** that the utility functions `get_auxiliary_data` and `load_to_dataframe` are implemented in [utils.py](utils.py).

This would return the optimal parameters as shown below for an AVX2 CPU if there is no sampling.
```
out_c_split_size                                                           3
out_r_split_size                                                           1
out_f_split_size                                                           4
```

## Step 7 - Create a Accera package with the optimal function
Finally, we can use the optimal parameters to create a Accera package with the best performant function. We can do ths by repeating [Step 2](#step-2---create-a-parameterized-accera-nchwc-2d-convolution-function) and replace the values in `parameters_values` by the optimal values.

## Pull it all together
For convenience, we wrote all the code snippets used in this case study in [run.py](run.py) and [utils.py](utils.py). To run all the case study steps, download the files and run:
```shell
python run.py --input_shape 7 7 512 --kernel_shape 3 3 --output_filters 512 --stride 1 1 --output_directory nchwc_conv2d_gridsearch_case_study --sample 100
```

> **_Note that_** the above command randomly selects 100 sample points out of the parameter grid for testing purposes. You can modify or remove the `--sample 100` argument to search fewer or more sample points.
