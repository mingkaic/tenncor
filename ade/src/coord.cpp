#include <cmath>

#include "ade/log/log.hpp"
#include "ade/matops.hpp"
#include "ade/coord.hpp"

#ifdef ADE_COORD_HPP

namespace ade
{

struct CoordMap final : public iCoordMap
{
	CoordMap (std::function<void(MatrixT)> init)
	{
		std::memset(fwd_, 0, mat_size);
		init(fwd_);
		inverse(bwd_, fwd_);
	}

	void forward (Shape::iterator out,
		Shape::const_iterator in) const override
	{
		std::array<double,rank_cap> temp;
		temp.fill(0);
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			for (uint8_t j = 0; j < rank_cap; ++j)
			{
				temp[j] += *(in + i) * fwd_[i][j];
			}
		}
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			out[i] = std::round(temp[i]);
		}
	}

	void backward (Shape::iterator out,
		Shape::const_iterator in) const override
	{
		std::array<double,rank_cap> temp;
		temp.fill(0);
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			for (uint8_t j = 0; j < rank_cap; ++j)
			{
				temp[j] += *(in + i) * bwd_[i][j];
			}
		}
		for (uint8_t i = 0; i < rank_cap; ++i)
		{
			out[i] = std::round(temp[i]);
		}
	}

	iCoordMap* reverse (void) const override
	{
		return new CoordMap(bwd_, fwd_);
	}

	std::string to_string (void) const override
	{
		std::stringstream ss;
		ss << arr_begin;
		for (uint8_t i = 0; i < rank_cap - 1; ++i)
		{
			ss << arr_begin << fwd_[i][0];
			for (uint8_t j = 1; j < rank_cap; ++j)
			{
				ss << arr_delim << fwd_[i][j];
			}
			ss << arr_end << arr_delim << '\n';
		}
		ss << arr_begin << fwd_[rank_cap - 1][0];
		for (uint8_t j = 1; j < rank_cap; ++j)
		{
			ss << arr_delim << fwd_[rank_cap - 1][j];
		}
		ss << arr_end << arr_end;
		return ss.str();
	}

private:
	CoordMap (const MatrixT fwd, const MatrixT bwd)
	{
		std::memcpy(fwd_, fwd, mat_size);
		std::memcpy(bwd_, bwd, mat_size);
	}

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

CoordPtrT reduce (uint8_t rank, std::vector<DimT> red)
{
	uint8_t n_red = red.size();
	if (std::any_of(red.begin(), red.end(),
		[](DimT& d) { return 0 == d; }))
	{
		fatalf("cannot reduce using zero dimensions %s",
			to_string(red).c_str());
	}
	if (rank + n_red > rank_cap)
	{
		fatalf("cannot reduce shape rank %d beyond rank_cap with n_red %d",
			rank, n_red);
	}
	if (0 == n_red)
	{
		warn("reducing with empty vector... created useless node");
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
		fatalf("cannot extend using zero dimensions %s",
			to_string(ext).c_str());
	}
	if (rank + n_ext > rank_cap)
	{
		fatalf("cannot extend shape rank %d beyond rank_cap with n_ext %d",
			rank, n_ext);
	}
	if (0 == n_ext)
	{
		warn("extending with empty vector... created useless node");
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
		warn("PERMUTING with same dimensions ... created useless node");
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

}

#endif
