#include "ade/matops.hpp"

#ifdef ADE_MATOPS_HPP

namespace ade
{

using AugMatrixT = double[mat_dim][mat_dim * 2];

// gauss_jordan_elim augmented matrix in-place
// return true if inversible
// algorithm taken from
// https://rosettacode.org/wiki/Gauss-Jordan_matrix_inversion#Go
static bool gauss_jordan_elim (AugMatrixT mat)
{
	RankT ncols = 2 * mat_dim;
	for (RankT row = 0, col = 0;
		row < mat_dim && col < ncols; ++row, ++col)
	{
		// search in submatrix [row:][col:] for next leading entry
		RankT next = row;
		while (col < ncols && mat[next][col] == 0)
		{
			if (mat_dim <= ++next)
			{
				next = row;
				++col;
			}
		}

		if (col >= mat_dim)
		{
			return false; // reduced (although non-inversible)
		}

		// assert(mat[next][col] != 0);
		std::swap(mat[next], mat[row]);
		// leading entry is now at [row][col]
		assert(mat[row][col] != 0);
		// row reduce by leading
		double leading = mat[row][col];
		for (RankT j = 0; j < ncols; ++j)
		{
			mat[row][j] /= leading;
		}

		// eliminate other rows by multiples of row
		for (RankT k = 0; k < mat_dim; ++k)
		{
			if (k != row)
			{
				double mult = mat[k][col];
				for (RankT j = 0; j < ncols; ++j)
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
	ss << fmts::arr_begin;
	for (RankT i = 0; i < mat_dim - 1; ++i)
	{
		ss << fmts::arr_begin << mat[i][0];
		for (RankT j = 1; j < mat_dim; ++j)
		{
			ss << fmts::arr_delim << mat[i][j];
		}
		ss << fmts::arr_end << fmts::arr_delim << '\n';
	}
	ss << fmts::arr_begin << mat[mat_dim - 1][0];
	for (RankT j = 1; j < mat_dim; ++j)
	{
		ss << fmts::arr_delim << mat[mat_dim - 1][j];
	}
	ss << fmts::arr_end << fmts::arr_end;
	return ss.str();
}

// calculate determinant using LU triangulation
// implementation taken from
// https://rosettacode.org/wiki/Determinant_and_permanent#C
double determinant (const MatrixT& mat)
{
	// copy mat over to temp array to allow row swapping
	double temp[mat_dim * mat_dim];
	std::memcpy(temp, mat, mat_size);
	double* m[mat_dim];
	m[0] = temp;
	for (RankT i = 1; i < mat_dim; i++)
	{
		m[i] = m[i - 1] + mat_dim;
	}

	// triangulate and collect reduced diagonals
	int sign = 1;
	double det = 1;
	for (RankT i = 0; i < mat_dim; i++)
	{
		RankT max_row = 0;
		for (RankT row = i; row < mat_dim; row++)
		{
			if (std::fabs(m[row][i]) > std::fabs(m[max_row][i]))
			{
				max_row = row;
			}
		}

		if (max_row > 0) // swap row
		{
			sign = -sign;
			double* tmp = m[i];
			m[i] = m[max_row];
			m[max_row] = tmp;
		}

		if (0 == m[i][i])
		{
			return 0;
		}
		det *= m[i][i];

		for (RankT row = i + 1; row < mat_dim; row++)
		{
			double r = m[row][i] / m[i][i];
			if (0 == r)
			{
				continue;
			}

			for (RankT col = i; col < mat_dim; col ++)
			{
				m[row][col] -= m[i][col] * r;
			}
		}
	}

	return det * sign;
}

void inverse (MatrixT out, const MatrixT& in)
{
	size_t rowbytes = sizeof(double) * mat_dim;
	AugMatrixT aug;
	for (RankT i = 0; i < mat_dim; ++i)
	{
		std::memcpy(aug[i], in[i], rowbytes);
		std::memset(aug[i] + mat_dim, 0, rowbytes);
		// augment by identity matrix to right
		aug[i][mat_dim + i] = 1;
	}
	if (false == gauss_jordan_elim(aug))
	{
		logs::fatalf("cannot invert matrix:\n%s", to_string(in).c_str());
	}
	// remove identity matrix to left
	for (RankT i = 0; i < mat_dim; ++i)
	{
		std::memcpy(out[i], aug[i] + mat_dim, rowbytes);
	}
}

void matmul (MatrixT out, const MatrixT& lhs, const MatrixT& rhs)
{
	for (RankT i = 0; i < mat_dim; ++i)
	{
		for (RankT j = 0; j < mat_dim; ++j)
		{
			out[i][j] = 0;
			for (RankT k = 0; k < mat_dim; ++k)
			{
				out[i][j] += lhs[i][k] * rhs[k][j];
			}
		}
	}
}

}

#endif
