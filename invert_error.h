#ifndef INVERT_ERROR_H
#define INVERT_ERROR_H

#include <string>
// I wish there was a more narrow header to import...
#include <CL/opencl.hpp>

/**
 * Retruns the name of the OpenCL error associated with the given error code.
 */
const std::string& invert_error(cl_int error_code);

#endif
