# Accera High Perf DL
High Performance Deep Learning using Accera

## Overview 
This repo presents some case studies for using [Accera](https://github.com/microsoft/Accera) - the open source cross-platform compiler from Microsoft Research - to create high performance deep learning computations (i.e. GEMM, Convolution, etc.).

## Contents
- [MatMul Grid Search Case Study](case_studies/matmul_gridsearch_on_avx2/README.md)
- [NCHWc 2D Convolution Grid Search Case Study](case_studies/nchwc_convolution_gridsearch/README.md)

## Documentation
Refer to the original [Accera Manual](https://microsoft.github.io/Accera/Manual/) to get familiar with Accera concepts and Domain Specific Language (DSL).

## Setup
Accera requires `Python 3.7-3.10`. For those case studies, you can simply install Accera using
```
pip install accera
```

To build Accera from the source, refer to Accera's [Install Instructions](https://microsoft.github.io/Accera/Install/).

> **_NOTE:_** Those case studies are tested on `Python 3.9.10` and `Accera v1.2.3`.

## Run the Case Studies
Each case study is located in a separate folder under the [case_studies](case_studies) folder. To run any case study, you can simply change directory to the required case study, then run
```
python run.py
```
Moreover, each case study has a README file to explain the details of what the case study is trying to do.

> **_NOTE:_** We will continiously add more case studies to show how to use Accera to build high performance deep learning computations.

## Contribute
To increase the impact of this repository, we welcome and encourage contributions with new case studies using Accera. Before you work on any, it is advised that you follow the same style as [MatMul Grid Search Case Study](case_studies/matmul_gridsearch_on_avx2) as follows:

1. Create `run.py` which contains the main code for the case study (mainly the Accera DSL).
2. Create `utils.py` which can include any needed utility functions.
3. Create `README.md` which contains a detailed explanation of what the case study is trying to achieve plus any results.

Following this template ensures that your work can be merged to the master branch in a timely manner.

## License 
MIT License. See [LICENSE](LICENSE) file

**_NOTE:_** For any questions about the case studies, feel free to reach out to me at <marina_neseem@brown.edu>.
