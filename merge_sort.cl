__kernel void merge(__global int* array, __global int* buffer, int start, int middle, int end) {
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

__kernel void merge_sort_bottom_up(__global int* array, __global int* buffer, int start, int end) {
  int single_step, double_step;
  for (single_step = 1, double_step = 2;
       start + double_step <= end;
       single_step *= 2, double_step *= 2) {
    int j;
    for (j = start; j + double_step <= end; j += double_step) {
      merge(array, buffer, j, j + single_step, j + double_step);
    }
    merge(array, buffer, j, j + single_step, end);
  }

  merge(array, buffer, start, start + single_step, end);
}
