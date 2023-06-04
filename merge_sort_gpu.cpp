#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cerrno>
#include <cstdlib>

#include <CL/opencl.hpp>

#include "invert_error.h"

struct State {
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
};

void compile_program(std::string program_filename, State& state) {
  cl_int error;
  // Step 1: Create an OpenCL context and command queue for the device
  state.context = cl::Context(CL_DEVICE_TYPE_GPU);
  std::vector<cl::Device> devices = state.context.getInfo<CL_CONTEXT_DEVICES>();
  std::cerr << "Number of GPUs detected: " << devices.size() << std::endl;
  state.queue = cl::CommandQueue(state.context, devices[0]);

  // Step 2: Load the OpenCL kernel source code and compile it into an OpenCL program
  std::ifstream file(program_filename);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string source = buffer.str();
  cl::Program::Sources sources(1, source.c_str());
  state.program = cl::Program(state.context, sources);
  cl_int build_status = state.program.build(devices);
  std::cerr << "Program build status: " << invert_error(build_status)
            << std::endl;
}

void merge(std::vector<int> &array, std::vector<int> &buffer, int start, int middle, int end) {
  int i = start, j = middle, k = 0;
  while (i < middle && j < end) {
    if (array[i] <= array[j])
      buffer[k++] = array[i++];
    else
      buffer[k++] = array[j++];
  }

  while (i < middle)
    buffer[k++] = array[i++];

  while (j < end)
    buffer[k++] = array[j++];

  for (int i = 0; i < k; ++i) {
    array[start + i] = buffer[i];
  }
}

void merge_sort_bottom_up(std::vector<int> &array, std::vector<int> &buffer, int start, int end, int initial_step) {
  int step, double_step;
  for (step = initial_step, double_step = initial_step * 2;
       start + double_step <= end;
       step *= 2, double_step *= 2) {
    int j;
    for (j = start; j + double_step <= end; j += double_step) {
      merge(array, buffer, j, j + step, j + double_step);
    }
    merge(array, buffer, j, j + step, end);
  }

  merge(array, buffer, start, start + step, end);
}

int main(int argc, char**argv) {
  State state;

  compile_program("merge_sort.cl", state);

  // Step 3: Create an OpenCL kernel object from the compiled program and the name of the kernel function
  cl::Kernel kernel(state.program, "merge_sort_bottom_up");

  // Step 4: Set the kernel arguments
  std::vector<int> array;
  int n;
  while (std::cin >> n) {
    array.push_back(n);
  }
  cl::Buffer array_message(state.context, CL_MEM_READ_WRITE, sizeof(int) * array.size());
  state.queue.enqueueWriteBuffer(array_message, CL_TRUE, 0, sizeof(int) * array.size(), array.data());

  std::vector<int> buffer(array.size());
  cl::Buffer buffer_message(state.context, CL_MEM_READ_WRITE, sizeof(int) * buffer.size());
  state.queue.enqueueWriteBuffer(buffer_message, CL_TRUE, 0, sizeof(int) * buffer.size(), buffer.data());

  kernel.setArg(0, array_message);
  kernel.setArg(1, buffer_message);

  cl::NDRange global_size(1);
  cl::NDRange local_size(1);


  int compute_units = 24;
  if (argc == 2) {
    char *endptr = NULL;
    long int arg = std::strtol(argv[1], &endptr, 10);
    if (arg > 0) {
      compute_units = static_cast<int>(arg);
    }
  }

  std::cerr << "Number of compute units: " << compute_units << std::endl;
  int chunk_size = array.size() / compute_units;

  auto chrono_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < compute_units; ++i) {
    kernel.setArg(2, i * chunk_size);
    kernel.setArg(3, (i + 1) * chunk_size);

    // Step 5: Enqueue the kernel for execution
    state.queue.enqueueNDRangeKernel(kernel, cl::NDRange(i), global_size, local_size);
  }

  // Step 6: Wait for the kernel to finish executing
  auto status = state.queue.finish();
  if (status != CL_SUCCESS) {
    std::cerr << "Computation status: " << invert_error(status) << std::endl;
  }

  // Step 7: Read the results back from the device
  std::vector<int> result(array.size());
  state.queue.enqueueReadBuffer(array_message, CL_TRUE, 0, sizeof(int) * array.size(), result.data());

  // Step 8: Do one last merge on the highest level

  auto chrono_end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>
    (chrono_end - chrono_start);

  // todo: measure
  merge_sort_bottom_up(result, buffer, 0, result.size(), chunk_size);

  std::cerr << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

  for (int el : result) {
    std::cout << el << std::endl;
  }

  return 0;
}
