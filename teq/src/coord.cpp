#include "teq/coord.hpp"

#ifdef TEQ_COORD_HPP

namespace teq
{

using WorkArrT = std::array<double,mat_dim>;

static inline void vecmul (WorkArrT& out,
	const MatrixT& mat, CoordT::const_iterator in)
{
	out.fill(0);
	for (RankT i = 0; i < mat_dim; ++i)
	{
		CDimT inv = 1;
		if (i < mat_dim - 1)
		{
			inv = *(in + i);
		}
		for (RankT j = 0; j < mat_dim; ++j)
		{
			out[j] += inv * mat[i][j];
		}
	}
}

void CoordMap::forward (CoordT::iterator out, CoordT::const_iterator in) const
{
	WorkArrT temp;
	vecmul(temp, fwd_, in);
	for (RankT i = 0; i < rank_cap; ++i)
	{
		out[i] = temp[i] / temp[rank_cap];
	}
}

CoordptrT identity(new CoordMap(
	[](MatrixT fwd)
	{
		for (RankT i = 0; i < rank_cap; ++i)
		{
			fwd[i][i] = 1;
		}
	}));

bool is_identity (iCoordMap* coorder)
{
	if (identity.get() == coorder || nullptr == coorder)
	{
		return true;
	}
	bool id = false;
	coorder->access([&id](const MatrixT& m)
	{
		id = true;
		for (RankT i = 0; id && i < mat_dim; ++i)
		{
			for (RankT j = 0; id && j < mat_dim; ++j)
			{
				id = id && m[i][j] == (i == j);
			}
		}
	});
	return id;
}

CoordptrT reduce (RankT rank, std::vector<DimT> red)
{
	RankT n_red = red.size();
	if (std::any_of(red.begin(), red.end(),
		[](DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot reduce using zero dimensions %s",
			fmts::to_string(red.begin(), red.end()).c_str());
	}
	if (rank + n_red > rank_cap)
	{
		logs::fatalf("cannot reduce shape rank %d beyond rank_cap with n_red %d",
			rank, n_red);
	}
	if (0 == n_red || std::all_of(red.begin(), red.end(),
		[](DimT d) { return 1 == d; }))
	{
		logs::warn("reducing scalar ... will do nothing");
		return identity;
	}

	return std::make_shared<CoordMap>(
		[&](MatrixT fwd)
		{
			for (RankT i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			for (RankT i = 0; i < n_red; ++i)
			{
				RankT outi = rank + i;
				fwd[outi][outi] = 1. / red[i];
			}
		});
}

CoordptrT extend (RankT rank, std::vector<DimT> ext)
{
	RankT n_ext = ext.size();
	if (std::any_of(ext.begin(), ext.end(),
		[](DimT& d) { return 0 == d; }))
	{
		logs::fatalf("cannot extend using zero dimensions %s",
			fmts::to_string(ext.begin(), ext.end()).c_str());
	}
	if (rank + n_ext > rank_cap)
	{
		logs::fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	if (0 == n_ext)
	{
		logs::warn("extending with empty vector ... will do nothing");
		return identity;
	}

	return std::make_shared<CoordMap>(
		[&](MatrixT fwd)
		{
			for (RankT i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			for (RankT i = 0; i < n_ext; ++i)
			{
				RankT outi = rank + i;
				fwd[outi][outi] = ext[i];
			}
		});
}

CoordptrT permute (std::vector<RankT> dims)
{
	if (dims.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return identity;
	}

	bool visited[rank_cap];
	std::memset(visited, false, rank_cap);
	for (RankT i = 0, n = dims.size(); i < n; ++i)
	{
		visited[dims[i]] = true;
	}
	for (RankT i = 0; i < rank_cap; ++i)
	{
		if (false == visited[i])
		{
			dims.push_back(i);
		}
	}

	return std::make_shared<CoordMap>(
		[&](MatrixT fwd)
		{
			for (RankT i = 0, n = dims.size(); i < n; ++i)
			{
				fwd[dims[i]][i] = 1;
			}
		});
}

CoordptrT flip (RankT dim)
{
	if (dim >= rank_cap)
	{
		logs::warn("flipping dimension out of rank_cap ... will do nothing");
		return identity;
	}

	return std::make_shared<CoordMap>(
		[&](MatrixT fwd)
		{
			for (RankT i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			fwd[dim][dim] = -1;
			fwd[rank_cap][dim] = -1;
		});
}

}

#endif
