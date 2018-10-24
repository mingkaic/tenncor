#include "ade/matops.hpp"
#include "ade/log.hpp"
#include "ade/coord.hpp"

#ifdef ADE_COORD_HPP

namespace ade
{

struct CoordMap final : public iCoordMap
{
	CoordMap (std::function<void(MatrixT)> init)
	{
		std::memset(fwd_, 0, rank_cap * rank_cap);
		init(fwd_);
		inverse(bwd_, fwd_);
	}

	void forward (Shape::iterator out, Shape::const_iterator in) const override
	{
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			for (uint8_t j = 0; j < rank_cap; ++j)
			{
				out[j] += *(in + i) * fwd_[i][j];
			}
		}
	}

	void backward (Shape::iterator out, Shape::const_iterator in) const override
	{
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			for (uint8_t j = 0; j < rank_cap; ++j)
			{
				out[j] += *(in + i) * bwd_[i][j];
			}
		}
	}

private:
	MatrixT fwd_;
	MatrixT bwd_;
};

CoordPtrT identity(new CoordMap(
	[](MatrixT fwd)
	{
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			fwd[i][i] = 1;
		}
	}));

CoordPtrT reduce (uint8_t dim)
{
	if (dim == 0)
	{
		warn("REDUCING coordinates [:0] ... created useless node");
		return identity();
	}

	return CoordPtrT(new CoordMap(
		[&](MatrixT fwd)
		{
			for (uint8_t i = 0, n = rank_cap - std::min(rank_cap, dim);
				i < n; ++i)
			{
				fwd[dim + i][i] = 1;
			}
		}));
}

CoordPtrT permute (std::vector<uint8_t> dims)
{
	if (dims.size() == 0)
	{
		warn("PERMUTING with same dimensions ... created useless node");
		return identity();
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
		});
}

CoordPtrT extend (uint8_t rank, std::vector<DimT> ext)
{
	uint8_t n_ext = ext.size();
	if (std::any_of(ext.begin(), ext.end(),
		[](DimT& d) { return 0 == d; }))
	{
		fatalf("cannot EXTEND using zero dimensions %s", to_string(ext));
	}
	if (rank + n_ext > rank_cap)
	{
		fatalf("cannot EXTEND shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	if (0 == n_ext)
	{
		warn("EXTENDing with empty vector... created useless node");
		return identity();
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

}

#endif
