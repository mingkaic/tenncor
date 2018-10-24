#include "ade/matops.hpp"

#ifdef ADE_MATOPS_HPP

namespace ade
{

void lu_decomposition (MatrixT lower, MatrixT upper, const MatrixT in)
{
    memset(lower, 0, sizeof(lower));
    memset(upper, 0, sizeof(upper));

    for (uint8_t i = 0; i < rank_cap; ++i)
	{
        // upper triangle
        for (uint8_t j = i; j < rank_cap; ++j)
		{
            // Σ(L[i][k] * U[k][j])
            double sum = 0;
            for (uint8_t k = 0; k < i; ++k)
			{
                sum += (lower[i][k] * upper[k][j]);
			}

            upper[i][j] = mat[i][j] - sum;
        }

        // lower triangle
        for (uint8_t j = i; j < rank_cap; ++j)
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

                lower[j][i] = (mat[j][i] - sum) / upper[i][i];
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
	for (uint8_t i = 0; i < rank_cap; ++i)
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
		for (uint8_t j = i + 1; j < rank_cap; ++j)
		{
			upper[i][j] /= upper[i][i];
		}
		upper[i][i] = 1 / upper[i][i];
	}
	// OUT = U^-1 @ L^-1
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		for (uint8_t j = 0; j < rank_cap; ++j)
		{
			out[i][j] = 0;
			for (uint8_t k = std::max(i, j); k < rank_cap; ++k)
			{
				out[i][j] += upper[i][k] * lower[k][j];
			}
		}
	}
}

}

#endif
