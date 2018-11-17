#include <cmath>

#include "err/log.hpp"
#include "ade/coord.hpp"

#ifdef ADE_COORD_HPP

namespace ade
{

using WorkArrT = std::array<double,mat_dim>;

static inline void vecmul (WorkArrT& out,
	const MatrixT& mat, CoordT::const_iterator in)
{
	out.fill(0);
	for (uint8_t i = 0; i < mat_dim; ++i)
	{
		ade::CDimT inv = 1;
		if (i < mat_dim - 1)
		{
			inv = *(in + i);
		}
		for (uint8_t j = 0; j < mat_dim; ++j)
		{
			out[j] += inv * mat[i][j];
		}
	}
}

void CoordMap::forward (CoordT::iterator out, CoordT::const_iterator in) const
{
	WorkArrT temp;
	vecmul(temp, fwd_, in);
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		out[i] = temp[i] / temp[rank_cap];
	}
}

void CoordMap::backward (CoordT::iterator out, CoordT::const_iterator in) const
{
	WorkArrT temp;
	vecmul(temp, bwd_, in);
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		out[i] = temp[i] / temp[rank_cap];
	}
}

CoordPtrT identity(new CoordMap(
	[](MatrixT fwd)
	{
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			fwd[i][i] = 1;
		}
	}));

CoordPtrT reduce (uint8_t rank, std::vector<DimT> red)
{
	uint8_t n_red = red.size();
	if (std::any_of(red.begin(), red.end(),
		[](DimT& d) { return 0 == d; }))
	{
		err::fatalf("cannot reduce using zero dimensions %s",
			err::to_string(red.begin(), red.end()).c_str());
	}
	if (rank + n_red > rank_cap)
	{
		err::fatalf("cannot reduce shape rank %d beyond rank_cap with n_red %d",
			rank, n_red);
	}
	if (0 == n_red)
	{
		err::warn("reducing with empty vector... will do nothing");
		return identity;
	}

	return CoordPtrT(new CoordMap(
		[&](MatrixT fwd)
		{
			for (uint8_t i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			for (uint8_t i = 0; i < n_red; ++i)
			{
				uint8_t outi = rank + i;
				fwd[outi][outi] = 1.0 / red[i];
			}
		}));
}

CoordPtrT extend (uint8_t rank, std::vector<DimT> ext)
{
	uint8_t n_ext = ext.size();
	if (std::any_of(ext.begin(), ext.end(),
		[](DimT& d) { return 0 == d; }))
	{
		err::fatalf("cannot extend using zero dimensions %s",
			err::to_string(ext.begin(), ext.end()).c_str());
	}
	if (rank + n_ext > rank_cap)
	{
		err::fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	if (0 == n_ext)
	{
		err::warn("extending with empty vector... will do nothing");
		return identity;
	}

	return CoordPtrT(new CoordMap(
		[&](MatrixT fwd)
		{
			for (uint8_t i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			for (uint8_t i = 0; i < n_ext; ++i)
			{
				uint8_t outi = rank + i;
				fwd[outi][outi] = ext[i];
			}
		}));
}

CoordPtrT permute (std::vector<uint8_t> dims)
{
	if (dims.size() == 0)
	{
		err::warn("permuting with same dimensions ... will do nothing");
		return identity;
	}

	bool visited[rank_cap];
	std::memset(visited, false, rank_cap);
	for (uint8_t i = 0, n = dims.size(); i < n; ++i)
	{
		visited[dims[i]] = true;
	}
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		if (false == visited[i])
		{
			dims.push_back(i);
		}
	}

	return CoordPtrT(new CoordMap(
		[&](MatrixT fwd)
		{
			for (uint8_t i = 0, n = dims.size(); i < n; ++i)
			{
				fwd[dims[i]][i] = 1;
			}
		}));
}

CoordPtrT flip (uint8_t dim)
{
	if (dim >= rank_cap)
	{
		err::warn("flipping dimension out of rank_cap ... will do nothing");
		return identity;
	}

	return CoordPtrT(new CoordMap(
		[&](MatrixT fwd)
		{
			for (uint8_t i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[dim][dim] = -1;
			fwd[rank_cap][dim] = -1;
		}));
}

}

#endif
