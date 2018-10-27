#include "ade/matops.hpp"

#ifdef ADE_MATOPS_HPP

namespace ade
{

void lu_decomposition (MatrixT lower, MatrixT upper, const MatrixT in)
{
	memset(lower, 0, mat_size);
	memset(upper, 0, mat_size);

	for (uint8_t i = 0; i < mat_dim; ++i)
	{
		// upper triangle
		for (uint8_t j = i; j < mat_dim; ++j)
		{
			// Σ(L[i][k] * U[k][j])
			double sum = 0;
			for (uint8_t k = 0; k < i; ++k)
			{
				sum += (lower[i][k] * upper[k][j]);
			}

			upper[i][j] = in[i][j] - sum;
		}

		// lower triangle
		for (uint8_t j = i; j < mat_dim; ++j)
		{
			if (i == j)
			{
				lower[i][i] = 1;
			}
			else
			{
				// Σ(L[j][k] * U[k][i])
				double sum = 0;
				for (uint8_t k = 0; k < i; ++k)
				{
					sum += (lower[j][k] * upper[k][i]);
				}

				lower[j][i] = (in[j][i] - sum) / upper[i][i];
			}
		}
	}
}

void inverse (MatrixT out, const MatrixT in)
{
	// IN = L @ U -> IN^-1 = U^-1 @ L^-1
	MatrixT lower;
	MatrixT upper;
	lu_decomposition(lower, upper, in);
	// inverse lower and upper
	for (uint8_t i = 0; i < mat_dim; ++i)
	{
		// lower triangle
		for (uint8_t j = 0; j < i; ++j)
		{
			lower[i][j] *= -1;
		}

		// upper triangle
		for (uint8_t j = 0; j < i; ++j)
		{
			upper[j][i] /= upper[i][i];
		}
		for (uint8_t j = i + 1; j < mat_dim; ++j)
		{
			upper[i][j] /= upper[i][i];
		}
		upper[i][i] = 1 / upper[i][i];
	}
	// OUT = U^-1 @ L^-1
	for (uint8_t i = 0; i < mat_dim; ++i)
	{
		for (uint8_t j = 0; j < mat_dim; ++j)
		{
			out[i][j] = 0;
			for (uint8_t k = std::max(i, j); k < mat_dim; ++k)
			{
				out[i][j] += upper[i][k] * lower[k][j];
			}
		}
	}
}

}

#endif
