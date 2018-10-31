#include "ade/matops.hpp"

#ifdef ADE_MATOPS_HPP

namespace ade
{

using AugMatrixT = double[mat_dim][mat_dim * 2];

// gauss_jordan_elim augmented matrix in-place
// return true if inversible
// algorithm taken from
// https://rosettacode.org/wiki/Gauss-Jordan_matrix_inversion#Go
// todo: simplify and clean up to fit C++ convention
bool gauss_jordan_elim (AugMatrixT mat)
{
	uint8_t ncols = 2 * mat_dim;
	for (uint8_t row = 0, col = 0;
		row < mat_dim && col < ncols; ++row, ++col)
	{
		// search in submatrix [row:][col:] for next leading entry
		uint8_t next = row;
		while (col < ncols && mat[next][col] == 0)
		{
			if (mat_dim == ++next)
			{
				next = row;
				++col;
			}
		}

		if (col >= ncols)
		{
			return false; // reduced (although non-inversible)
		}

		std::swap(mat[next], mat[row]);
		// leading entry is now at [row][col]
		double div = mat[row][col];
		if (div != 0)
		{
			for (uint8_t j = 0; j < ncols; ++j)
			{
				mat[row][j] /= div;
			}
		}

		for (uint8_t k = 0; k < mat_dim; ++k)
		{
			if (k != row)
			{
				double mult = mat[k][col];
				for (uint8_t j = 0; j < ncols; ++j)
				{
					mat[k][j] -= mat[row][j] * mult;
				}
			}
		}
	}
	return true;
}

std::string to_string (const MatrixT& mat)
{
	std::stringstream ss;
	ss << arr_begin;
	for (uint8_t i = 0; i < mat_dim - 1; ++i)
	{
		ss << arr_begin << mat[i][0];
		for (uint8_t j = 1; j < mat_dim; ++j)
		{
			ss << arr_delim << mat[i][j];
		}
		ss << arr_end << arr_delim << '\n';
	}
	ss << arr_begin << mat[mat_dim - 1][0];
	for (uint8_t j = 1; j < mat_dim; ++j)
	{
		ss << arr_delim << mat[mat_dim - 1][j];
	}
	ss << arr_end << arr_end;
	return ss.str();
}

void inverse (MatrixT out, const MatrixT& in)
{
	size_t rowbytes = sizeof(double) * mat_dim;
	AugMatrixT aug;
	for (uint8_t i = 0; i < mat_dim; ++i)
	{
		std::memcpy(aug[i], in[i], rowbytes);
		std::memset(aug[i] + mat_dim, 0, rowbytes);
		// augment by identity matrix to right
		aug[i][mat_dim + i] = 1;
	}
	if (false == gauss_jordan_elim(aug))
	{
		fatalf("cannot invert matrix:\n%s", to_string(in).c_str());
	}
	// remove identity matrix to left
	for (uint8_t i = 0; i < mat_dim; ++i)
	{
		std::memcpy(out[i], aug[i] + mat_dim, rowbytes);
	}
}

}

#endif
