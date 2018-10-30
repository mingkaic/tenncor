#include "ade/matops.hpp"

#ifdef ADE_MATOPS_HPP

namespace ade
{

using AugMatrixT = double[mat_dim][mat_dim * 2];

// reduce row echelon form mat augmented matrix in-place
// algorithm taken from
// https://rosettacode.org/wiki/Gauss-Jordan_matrix_inversion#Go
// todo: simplify and clean up to fit C++ convention
static inline void rrow_echelon_form (AugMatrixT mat)
{
	uint8_t lead = 0;
	for (uint8_t r = 0; r < mat_dim; r++)
	{
		if (2 * mat_dim <= lead)
		{
			return;
		}
		uint8_t i = r;

		while (mat[i][lead] == 0)
		{
			i++;
			if (mat_dim == i)
			{
				i = r;
				lead++;
				if (2 * mat_dim == lead)
				{
					return;
				}
			}
		}

		std::swap(mat[i], mat[r]);
		double div = mat[r][lead];
		if (div != 0)
		{
			for (uint8_t j = 0; j < 2 * mat_dim; ++j)
			{
				mat[r][j] /= div;
			}
		}

		for (uint8_t k = 0; k < mat_dim; ++k)
		{
			if (k != r)
			{
				double mult = mat[k][lead];
				for (uint8_t j = 0; j < 2 * mat_dim; ++j)
				{
					mat[k][j] -= mat[r][j] * mult;
				}
			}
		}
		lead++;
	}
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
	rrow_echelon_form(aug);
	// remove identity matrix to left
	for (uint8_t i = 0; i < mat_dim; ++i)
	{
		std::memcpy(out[i], aug[i] + mat_dim, rowbytes);
	}
}

}

#endif
