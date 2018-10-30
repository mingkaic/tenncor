///
///	matops.hpp
///	ade
///
///	Purpose:
///	Define matrix operations for coordinate transformation
/// This functions are here to avoid external dependencies in ADE
///

#include <cstring>

#include "ade/shape.hpp"

#ifndef ADE_MATOPS_HPP
#define ADE_MATOPS_HPP

namespace ade
{

const uint8_t mat_dim = rank_cap + 1; // todo: make homogeneous (for large shapes)

const size_t mat_size = sizeof(double) * mat_dim * mat_dim;

using MatrixT = double[mat_dim][mat_dim];

std::string to_string (const MatrixT& mat);

/// Inverse in matrix and dump to out matrix
void inverse (MatrixT out, const MatrixT& in);

}

#endif /// ADE_MATOPS_HPP
