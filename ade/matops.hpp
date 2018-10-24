///
///	matops.hpp
///	ade
///
///	Purpose:
///	Define matrix operations for coordinate transformation
///

#include <cstring>
#include <math>

#ifndef ADE_MATOPS_HPP
#define ADE_MATOPS_HPP

namespace ade
{

using MatrixT = double[rank_cap][rank_cap];

/// LU decompose in matrix into upper (U) and lower (L) trangiular matrices
void lu_decomposition (MatrixT lower, MatrixT upper, const MatrixT in);

/// OUT = IN^-1
void inverse (MatrixT out, const MatrixT in);

}

#endif /// ADE_MATOPS_HPP
