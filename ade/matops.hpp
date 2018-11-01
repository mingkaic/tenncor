///
///	matops.hpp
///	ade
///
///	Purpose:
///	Define matrix operations for coordinate transformation
/// This functions are here to avoid external dependencies in ADE
///

#include "ade/shape.hpp"

#ifndef ADE_MATOPS_HPP
#define ADE_MATOPS_HPP

namespace ade
{

/// Number of rows and columns for the homogeneous matrix
const uint8_t mat_dim = rank_cap + 1;

/// Number of bytes in a homogeneous matrix
const size_t mat_size = sizeof(double) * mat_dim * mat_dim;

/// Coordinate transformation matrix (using homogeneous)
using MatrixT = double[mat_dim][mat_dim];

/// Return the string representation of input matrix
std::string to_string (const MatrixT& mat);

/// Inverse in matrix and dump to out matrix
void inverse (MatrixT out, const MatrixT& in);

}

#endif /// ADE_MATOPS_HPP
