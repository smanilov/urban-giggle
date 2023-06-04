#!/bin/bash

g++ -DCL_HPP_TARGET_OPENCL_VERSION=300 -c invert_error.cpp
g++ -DCL_HPP_TARGET_OPENCL_VERSION=300 -lOpenCL invert_error.o merge_sort_gpu.cpp
