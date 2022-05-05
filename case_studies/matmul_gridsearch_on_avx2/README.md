[//]: # (Project: Accera)
[//]: # (Version: 1.2.3)

# Case Study - MatMul Grid Search

In this case study, we will discuss how to construct a performant implementation for matrix multiplication on an AVX2 Machine using [Accera](https://microsoft.github.io/Accera/). First, we will show how to create a parameterized Accera schedule and plan, then we will discuss how to create a parameters grid, and how to benchmark each point in the grid in order to pick the best performant implementation. Note that for different target hardwares, the process would be similar, however, the results would be different.

As introduced in [Section 0](https://microsoft.github.io/Accera/Manual/00%20Introduction) of the Accera Manual, a matrix multiplication between an `M`&times;`S` matrix `A` and an `S`&times;`N` matrix `B`, will result in an `M`&times;`N` matrix `C`, where `C += A @ B`. The logic for the matrix multiplication can be expressed in python as

```python
for i in range(M):
    for j in range(N):
        for k in range(S):
            C[i,j] += A[i,k] * B[k,j]
```

In this case study, we present the end-to-end steps needed to write a performant implementation for matrix multiplication as follows:

- [Step 1 - Create a Standard Accera Matmul Function](#step-1---create-a-standard-accera-matmul-function)
- [Step 2 - Create a Parmeterized Accera Matmul Function](#step-2---create-a-parameterized-accera-matmul-function)
- [Step 3 - Create a parameters grid](#step-3---create-a-parameters-grid)
- [Step 4 - Define a filter function to filter the parameters grid](#step-4---define-a-filter-function-to-filter-the-parameters-grid)
- [Step 5 - Create a Accera package with all parameters choices in the filtered parameters grid](#step-5---create-a-accera-package-with-all-parameters-choices-in-the-filtered-parameters-grid)
- [Step 6 - Benchmark the package on the target hardware](#step-6---benchmark-the-package-on-the-target-hardware)
- [Step 7 - Find the optimal parameters choice](#step-7---find-the-optimal-parameters-choice)
- [Step 8 - Create a Accera package with the optimal function](#step-8---create-a-accera-package-with-the-optimal-function)
- [Pull it all together](#pull-it-all-together)


## Step 1 - Create a Standard Accera Matmul Function
As introduced in [Section 2](https://microsoft.github.io/Accera/Manual/02%20Simple%20Affine%20Loop%20Nests/) of the Accera manual, we can implement the above python implementation for matrix multiplication using Accera as follows:

```python
import accera as acc

M, N, S = 1020, 1024, 1024

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

# Define a simple affine loop nest and name its loops i, j, k
nest = acc.Nest(shape=(M, N, S))
i, j, k = nest.get_indices()

# Define the logic of each iteration in the nest
@nest.iteration_logic
def _():
    C[i,j] += A[i,k] * B[k,j]

# Define Schedule
schedule = nest.create_schedule()

# Define Plan
plan = schedule.create_plan()

# Add the function to a package and build the package
name = f"matmul_{M}_{N}_{S}"
package = acc.Package()
function = package.add(plan, args=(A, B, C), base_name=name)
package.build(name, format=acc.Package.Format.HAT_DYNAMIC, output_dir=name)
```

This implementation would use the default schedule and plan. This Accera snippet would produce the correct results, but will not produce a high performance implementation. We can improve the efficiency of this Accera implementation using schedule and plan optimizations such as splits, vectorization, unrolling, and caching as we will do in the next step.

## Step 2 - Create a Parameterized Accera Matmul Function
For this case study, we choose to first do a tiling split per dimension to fit some active tiles of the input and output matrices in the L2 Cache, then we will do a second round of smaller splits, unrolling and vectorization along some dimensions to make use of the registers on the target hardware.

For this step, we will start with some chosen values for the split sizes and the schedule loop order. However, in the next steps we will discuss how to define a parameter grid and search through it to get the correct parameters for the target hardware.

First, we define the input and output arrays
```python
A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))
```

Then, we define some parameters that would be used later while creating the schedule and the plan
```python
p_m_split_size, p_n_split_size, p_s_split_size, \
        p_s_split_2_size, p_n_split_2_size, p_n_split_3_size = acc.create_parameters(6)
```

Then, we define the iteration logic for matrix multiplication
```python
# Define a simple affine loop nest and name its loops i, j, k
nest = acc.Nest(shape=(M, N, S))
i, j, k = nest.get_indices()

# Define the logic of each iteration in the nest
@nest.iteration_logic
def _():
    C[i,j] += A[i,k] * B[k,j]
```

Next, we define the matrix multiplication schedule. As mentioned earlier, we choose to first do a tiling split per dimension to improve cache utilization, then we do a second round of smaller splits to improve register utilization.
```python
schedule = nest.create_schedule()

# Tile splits to place some blocks of the input and output matrices in the L2 cache
ii, jj, kk = schedule.tile({i: p_m_split_size, j: p_n_split_size, k: p_s_split_size})

# Kernel splits
kkk = schedule.split(kk, p_s_split_2_size)
jjj = schedule.split(jj, p_n_split_2_size)
jjjj = schedule.split(jjj, p_n_split_3_size)

# Apply re-ordering
schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)
```

Then, we define a plan, and we use caching, unrolling, and vectorization to make the implementation more performant.
```python
plan = schedule.create_plan()

# Cache input and output arrays
plan.cache(B, index=jj)
plan.cache(C, index=ii)

# Unroll the non-vectorized kernel loops
plan.unroll(ii)
plan.unroll(jjj)

# Vectorize the innermost kernel loop
plan.vectorize(jjjj)
```
Then, we create a package with this matrix multiplication function.
```python
package = acc.Package()
name = f"matmul_{M}_{N}_{S}"
parameters_values = {p_m_split_size:6, p_n_split_size:256, p_s_split_size:128, \
                     p_s_split_2_size:16, p_n_split_2_size:8, p_n_split_3_size:4}
function = package.add(plan, args=(A, B, C), parameters=parameters_values, base_name=name)
package.build(name, format=acc.Package.Format.HAT_DYNAMIC, output_dir=name)
```

Pulling it all together
```python
import accera as acc

M, N, S = 1020, 1024, 1024

A = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(M, S))
B = acc.Array(role=acc.Array.Role.INPUT, element_type=acc.ScalarType.float32, shape=(S, N))
C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, element_type=acc.ScalarType.float32, shape=(M, N))

p_m_split_size, p_n_split_size, p_s_split_size, \
        p_s_split_2_size, p_n_split_2_size, p_n_split_3_size = acc.create_parameters(6)

# Define a simple affine loop nest and name its loops i, j, k
nest = acc.Nest(shape=(M, N, S))
i, j, k = nest.get_indices()

# Define the logic of each iteration in the nest
@nest.iteration_logic
def _():
    C[i,j] += A[i,k] * B[k,j]

schedule = nest.create_schedule()

# Tile splits to place some blocks of the input and output matrices in the L2 cache
ii, jj, kk = schedule.tile({i: p_m_split_size, j: p_n_split_size, k: p_s_split_size})

# Kernel splits
kkk = schedule.split(kk, p_s_split_2_size)
jjj = schedule.split(jj, p_n_split_2_size)
jjjj = schedule.split(jjj, p_n_split_3_size)

# Apply re-ordering
schedule.reorder(j, k, i, jj, kk, kkk, ii, jjj, jjjj)

plan = schedule.create_plan()

# Cache input and output arrays
plan.cache(B, index=jj)
plan.cache(C, index=ii)

# Unroll the non-vectorized kernel loops
plan.unroll(ii)
plan.unroll(jjj)

# Vectorize the innermost kernel loop
plan.vectorize(jjjj)

package = acc.Package()
name = f"matmul_{M}_{N}_{S}"
parameters_values = {p_m_split_size:6, p_n_split_size:256, p_s_split_size:128, \
                     p_s_split_2_size:16, p_n_split_2_size:8, p_n_split_3_size:4}
function = package.add(plan, args=(A, B, C), parameters=parameters_values, base_name=name)
package.build(name, format=acc.Package.Format.HAT_DYNAMIC, output_dir=name)
```

## Step 3 - Create a parameters grid
As mentioned in the previous step, we assumed some values for the split sizes. However, we are not sure that those chosen sizes would give the best performance, also for each different hardware the parameters values that achieves the best performant implementation can be different. To ensure that the created Accera function is performant (i.e. has the right parameters), we define a parameters grid where our chosen parameters are:

1. `p_m_split_size`
2. `p_n_split_size`
3. `p_s_split_size`
4. `p_n_split_2_size`
5. `p_n_split_3_size`
6. `p_s_split_2_size`

and our grid will consist of a set of possible values for those parameters.

For example, we might want to:

1. define the `p_m_split_size`, `p_n_split_size`, and `p_s_split_size` split sizes as the even numbers between 4 and 8,
   and the powers of 2 between 16 and 256.
2. define the `p_n_split_2_size`, `p_n_split_3_size`, and `p_s_split_2_size` as the powers of 2 between 4 and 16.

We can create our parameters grid by calling create_parameter_grid method.

```python
parameters_list = acc.create_parameter_grid({
    p_m_split_size: [4, 6, 8, 16, 32, 64, 128, 256],
    p_n_split_size: [4, 6, 8, 16, 32, 64, 128, 256], 
    p_s_split_size: [4, 6, 8, 16, 32, 64, 128, 256],
    p_s_split_2_size: [4, 8, 16],
    p_n_split_2_size: [4, 8, 16],
    p_n_split_3_size: [4, 8, 16]           
    })
```

## Step 4 - Define a filter function to filter the parameters grid
The previous step would produce a **large parameters grid**. For example, the chosen parameters choices in the previous step results in a parameters grid of size **13,824**, if each different point takes 5 second of evaluation, then this run would take **19.2 hours**. This can be a big overhead. However, we can notice that some points in the parameters grid might not be worth searching. For example:

1. It is meaningless to choose a second split of size larger than the first split on the same dimension, so we can filter those cases out.
2. We know that we need the active tiles of the input and output matrices to fit in the L2 cache without overflowing it, so we can choose the tile sizes such that the total memory needed for the active tiles of the input and output matrices are at least 50\% of the cache size, but less that its total size.

Using those simple filters, we can reduce the parameters grid size, hence reduce the time needed for evaluation.

```python
def filter_function(parameters_choice):
    l2_cache_size = 256
    element_size = 4
    m_split_size, n_split_size, s_split_size, s_split_2_size, n_split_2_size, n_split_3_size = parameters_choice

    return valid_split_size(n_split_size, n_split_2_size) \
        and valid_split_size(n_split_2_size, n_split_3_size) \
        and valid_split_size(s_split_size, s_split_2_size) \
        and fits_in_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size) \
        and uses_enough_l2(m_split_size, n_split_size, s_split_size, element_size, l2_cache_size)
```
> **_Note_** that the utility functions `valid_split_size`, `fits_in_l2`, and `uses_enough_l2` are implemented in [utils.py](utils.py).

and that filter function can be applied to the parameters grid using 
```python
parameters_list = acc.create_parameter_grid({
    p_m_split_size: [4, 6, 8, 16, 32, 64, 128, 256],
    p_n_split_size: [4, 6, 8, 16, 32, 64, 128, 256], 
    p_s_split_size: [4, 6, 8, 16, 32, 64, 128, 256],
    p_s_split_2_size: [4, 8, 16],
    p_n_split_2_size: [4, 8, 16],
    p_n_split_3_size: [4, 8, 16]           
    }, filter_func=filter_function, sample=10)
```
> **_Note_** `sample` can be used to specify the size of a random sample from the parameters grid (for testing purposes).


## Step 5 - Create a Accera package with all parameters choices in the filtered parameters grid
In this step, we would create a Accera package with all parameters choices and the filter defined above.

This can be done simply by replacing the below code from [Step 2](#step-2---create-a-parameterized-accera-matmul-function)
```python
function = package.add(plan, args=(A, B, C), parameters=parameters_values, base_name=f"matmul_{M}_{N}_{S}")
```
to 
```python
function = package.add(plan, args=(A, B, C), parameters=parameters_list, base_name=f"matmul_{M}_{N}_{S}")
```
where `parameters_list` is defined in the previous step.

## Step 6 - Benchmark the package on the target hardware
Finally, we can use [HAT tools](https://github.com/microsoft/hat) to benchmark our Accera package on the target hardware and write back the timing results to the package.
```python
hat_file_path = os.path.join(package_directory, "matmul.hat")
hat.run_benchmark(hat_file_path, store_in_hat=write_back, batch_size=5)
```

## Step 7 - Find the optimal parameters choice
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
    return [[int(optimal_point["p_m_split_size"])],
            [int(optimal_point["p_n_split_size"])],
            [int(optimal_point["p_s_split_size"])],
            [int(optimal_point["p_s_split_2_size"])],
            [int(optimal_point["p_n_split_2_size"])],
            [int(optimal_point["p_n_split_3_size"])]]
            
```
> **_Note_** that the utility functions `get_auxiliary_data` and `load_to_dataframe` are implemented in [utils.py](utils.py).

This would return the optimal parameters as shown below for an AVX2 CPU if there is no sampling.
```
m_split_size                                           6
n_split_size                                         256
s_split_size                                         128
n_split_2_size                                        16
n_split_3_size                                         8
s_split_2_size                                         4
```

## Step 8 - Create a Accera package with the optimal function
Finally, we can use the optimal parameters to create a Accera package with the best performant function. We can do ths by repeating [Step 2](#step-2---create-a-parameterized-accera-matmul-function) and replace the values in `parameters_values` by the optimal values.


## Pull it all together
For convenience, we wrote all the code snippets used in this case study in [run.py](run.py) and [utils.py](utils.py). To run all the case study steps, download the files and run:
```shell
python run.py --matmul_dim 1020 1024 1024 --output_directory matmul_gridsearch_case_study --sample 100
```

> **_Note that_** the above command randomly selects 100 sample points out of the parameter grid for testing purposes. You can modify or remove the `--sample 100` argument to search fewer or more sample points.
