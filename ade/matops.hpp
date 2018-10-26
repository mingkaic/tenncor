///
///	matops.hpp
///	ade
///
///	Purpose:
///	Define matrix operations for coordinate transformation
///

#include <cstring>

#include "ade/shape.hpp"

#ifndef ADE_MATOPS_HPP
#define ADE_MATOPS_HPP

namespace ade
{

const uint8_t mat_dim = rank_cap; // todo: make homogeneous

const size_t mat_size = sizeof(double) * mat_dim * mat_dim;

using MatrixT = double[mat_dim][mat_dim];

/// LU decompose in matrix into upper (U) and lower (L) trangiular matrices
void lu_decomposition (MatrixT lower, MatrixT upper, const MatrixT in);

/// OUT = IN^-1
void inverse (MatrixT out, const MatrixT in);

}

#endif /// ADE_MATOPS_HPP
