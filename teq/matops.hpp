///
/// matops.hpp
/// teq
///
/// Purpose:
/// Define matrix operations for coordinate transformation
/// This functions are here to avoid external dependencies in TEQ
///

#include <cassert>
#include <cstring>

#include "teq/shape.hpp"

#ifndef TEQ_MATOPS_HPP
#define TEQ_MATOPS_HPP

namespace teq
{

/// Number of rows and columns for the homogeneous matrix
const RankT mat_dim = rank_cap + 1;

/// Number of bytes in a homogeneous matrix
const size_t mat_size = mat_dim * mat_dim;

/// Coordinate transformation matrix (using homogeneous)
using MatrixT = double[mat_dim][mat_dim];

/// Return the string representation of input matrix
std::string to_string (const MatrixT& mat);

/// Return the determinant of matrix
double determinant (const MatrixT& mat);

/// Inverse in matrix and dump to out matrix
void inverse (MatrixT out, const MatrixT& in);

/// Apply matrix multiplication for lhs and rhs to out matrix
void matmul (MatrixT out, const MatrixT& lhs, const MatrixT& rhs);

}

#endif /// TEQ_MATOPS_HPP
