#include "teq/coord.hpp"

#ifdef TEQ_COORD_HPP

namespace teq
{

using WorkArrT = std::array<CDimT,mat_dim>;

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

Shape ShapeMap::convert (const Shape& shape) const
{
	CoordT in;
	std::copy(shape.begin(), shape.end(), in.begin());
	WorkArrT temp;
	vecmul(temp, fwd_, in.begin());
	std::vector<DimT> slist;
	slist.reserve(rank_cap);
	for (RankT i = 0; i < rank_cap; ++i)
	{
		auto cd = temp[i] / temp[rank_cap];
		if (cd < 0)
		{
			cd = -cd - 1;
		}
		slist.push_back(std::ceil(cd));
	}
	return Shape(slist);
}

ShaperT identity(new ShapeMap(
	[](MatrixT& fwd)
	{
		for (RankT i = 0; i < rank_cap; ++i)
		{
			fwd[i][i] = 1;
		}
	}));

bool is_identity (ShapeMap* shaper)
{
	if (identity.get() == shaper || nullptr == shaper)
	{
		return true;
	}
	bool id = false;
	shaper->access([&id](const MatrixT& m)
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

ShaperT reduce (std::set<RankT> rdims)
{
	size_t n_red = rdims.size();
	if (std::any_of(rdims.begin(), rdims.end(),
		[](RankT d) { return d >= teq::rank_cap; }))
	{
		logs::fatalf(
			"cannot reduce using dimensions greater or equal to rank_cap: %s",
			fmts::to_string(rdims.begin(), rdims.end()).c_str());
	}
	if (n_red > rank_cap)
	{
		logs::fatalf("cannot reduce %d rank when only ranks are capped at %d",
			n_red, rank_cap);
	}
	return std::make_shared<ShapeMap>(
		[&](MatrixT& fwd)
		{
			for (RankT i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = 1;
			}
			for (RankT i : rdims)
			{
				fwd[i][i] = std::numeric_limits<CDimT>::min();
			}
		});
}

ShaperT extend (teq::CoordT bcast)
{
	if (std::any_of(bcast.begin(), bcast.end(),
		[](RankT rank) { return rank < 1; }))
	{
		logs::fatalf("cannot extend with zero values: %s",
			fmts::to_string(bcast.begin(), bcast.end()).c_str());
	}

	return std::make_shared<ShapeMap>(
		[&](MatrixT& fwd)
		{
			for (RankT i = 0; i < rank_cap; ++i)
			{
				fwd[i][i] = bcast[i];
			}
		});
}

ShaperT permute (std::array<RankT,rank_cap> order)
{
	if (std::any_of(order.begin(), order.end(),
		[](RankT i) { return i >= rank_cap; }))
	{
		logs::fatalf("cannot permute with ranks greater than cap: %s",
			fmts::to_string(order.begin(), order.end()).c_str());
	}
	if (std::set<RankT>(order.begin(), order.end()).size() < rank_cap)
	{
		logs::fatalf("permute does not support repeated orders: %s",
			fmts::to_string(order.begin(), order.end()).c_str());
	}
	return std::make_shared<ShapeMap>(
		[&](MatrixT& fwd)
		{
			for (RankT i = 0, n = order.size(); i < n; ++i)
			{
				fwd[order[i]][i] = 1;
			}
		});
}

ShaperT reduce (RankT rank, std::vector<DimT> red)
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

	return std::make_shared<ShapeMap>(
		[&](MatrixT& fwd)
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

ShaperT extend (RankT rank, std::vector<DimT> ext)
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

	return std::make_shared<ShapeMap>(
		[&](MatrixT& fwd)
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

ShaperT permute (std::vector<RankT> dims)
{
	if (dims.size() == 0)
	{
		logs::warn("permuting with same dimensions ... will do nothing");
		return identity;
	}

	bool visited[rank_cap];
	std::fill(visited, visited + rank_cap, false);
	for (RankT i = 0, n = dims.size(); i < n; ++i)
	{
		if (visited[dims[i]])
		{
			logs::fatalf("permute does not support repeated orders: %s",
				fmts::to_string(dims.begin(), dims.end()).c_str());
		}
		visited[dims[i]] = true;
	}
	for (RankT i = 0; i < rank_cap; ++i)
	{
		if (false == visited[i])
		{
			dims.push_back(i);
		}
	}

	return std::make_shared<ShapeMap>(
		[&](MatrixT& fwd)
		{
			for (RankT i = 0, n = dims.size(); i < n; ++i)
			{
				fwd[dims[i]][i] = 1;
			}
		});
}

}

#endif
