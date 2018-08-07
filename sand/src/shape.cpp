#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>

#include "sand/shape.hpp"
#include "util/error.hpp"

#ifdef SAND_SHAPE_HPP

Shape::Shape (void) : rank_(0)
{
	std::fill(dims_.begin(), dims_.end(), 1);
}

Shape::Shape (std::vector<DimT> dims) :
	rank_(rank_cap)
{
	vector_assign(dims);
}

Shape::Shape (std::vector<Shape> shapes) :
	groups_(0), rank_(0)
{
	auto dest_it = dims_.begin();
	for (uint8_t i = 0, n = shapes.size(); i < n
		&& rank_ + shapes[i].n_rank() < rank_cap; ++i)
	{
		auto src_it = shapes[i].dims_.begin();
		uint8_t rank = shapes[i].n_rank();
		std::copy(src_it, src_it + rank, dest_it + rank_);
		rank_ += rank;
		// [lsb] 0 ... rank_cap-1 [msb]
		groups_ |= 1 << (rank_ - 1); // set group delimiting ranks
	}
	groups_ |= 0xff << (rank_);
	std::fill(dest_it + rank_, dims_.end(), 1);
}

Shape::Shape (const Shape& other) = default;

Shape& Shape::operator = (const Shape& other) = default;

Shape& Shape::operator = (const std::vector<DimT>& dims)
{
	vector_assign(dims);
	return *this;
}

Shape::Shape (Shape&& other)
{
	move_helper(std::move(other));
}

Shape& Shape::operator = (Shape&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other));
	}
	return *this;
}

DimT Shape::at (uint8_t idx) const
{
	if (idx >= rank_cap)
	{
		throw std::out_of_range(
			"accessing dimension out of allocated rank cap");
	}
	return dims_.at(idx);
}

Shape Shape::group (uint8_t idx) const
{
	// access first index of idx-th group
	uint8_t i = kth_group(idx);
	if (i >= rank_cap)
	{
		throw std::out_of_range(
			"accessing group out of allocated rank cap");
	}

	std::vector<DimT> out;
	for (bool go = true; i < rank_cap && go; ++i)
	{
		out.push_back(dims_[i]);
		go = 0 == ((groups_ >> i) & 1);
	}
	return Shape(out);
}

uint8_t Shape::n_rank (void) const
{
	return rank_;
}

NElemT Shape::n_elems (void) const
{
	auto it = dims_.begin();
	return std::accumulate(it, it + rank_, 1,
		std::multiplies<NElemT>());
}

std::vector<DimT> Shape::as_list (void) const
{
	auto it = dims_.begin();
	return std::vector<DimT>(it, it + rank_);
}

std::vector<Shape> Shape::as_groups (void) const
{
	std::vector<Shape> out;
	std::vector<DimT> slist;
	for (uint8_t i = 0; i < rank_; ++i)
	{
		slist.push_back(dims_[i]);
		if (0 < ((groups_ >> i) & 1))
		{
			out.push_back(Shape(slist));
			slist = {};
		}
	}
	if (false == slist.empty())
	{
		out.push_back(Shape(slist));
	}
	return out;
}

bool Shape::compatible_before (const Shape& other, uint8_t idx) const
{
	auto it = dims_.begin();
	uint8_t cap = rank_cap;
	return std::equal(it,
		it + std::min(idx, cap),
		other.dims_.begin());
}

bool Shape::compatible_after (const Shape& other, uint8_t idx) const
{
	bool compatible = false;
	if (idx < rank_cap)
	{
		compatible = std::equal(dims_.begin() + idx,
			dims_.end(), other.dims_.begin() + idx);
	}
	return compatible;
}

std::string Shape::to_string (void) const
{
	std::stringstream ss;
	bool group_done = groups_ & 1;
	ss << (unsigned) dims_[0];
	for (uint8_t i = 1; i < rank_; ++i)
	{
		ss << (group_done ? " " : ",") << (unsigned) dims_[i];
		group_done = (groups_ >> i) & 1;
	}
	return ss.str();
}

Shape::ShapeIterator Shape::begin (void)
{
	return dims_.begin();
}

Shape::ShapeIterator Shape::end (void)
{
	return dims_.end();
}

void Shape::vector_assign (const std::vector<DimT>& dims)
{
	auto src = dims.begin();
	if (std::any_of(src, dims.end(),
		[](DimT d)
		{
			return d == 0;
		}))
	{
		handle_error("shape assignment with zero vector",
			ErrVec<DimT>{"vec", dims});
	}
	auto dest = dims_.begin();

	uint8_t newrank = std::min((size_t) rank_cap, dims.size());
	std::copy(src, src + newrank, dest);
	if (newrank < rank_)
	{
		std::fill(dest + newrank, dest + rank_, 1);
	}
	rank_ = newrank;
}

void Shape::move_helper (Shape&& other)
{
	dims_ = std::move(other.dims_);
	rank_ = std::move(other.rank_);
	groups_ = std::move(other.groups_);

	auto ot = other.dims_.begin();
	std::fill(ot, ot + rank_, 1);
	other.rank_ = 0;
}

uint8_t Shape::kth_group (uint8_t k) const
{
	// [lsb] 0 ... rank_cap-1 [msb]
	uint8_t i = 0;
	for (; i < rank_cap && k > 0; ++i)
	{
		k -= (groups_ >> i) & 1;
	}
	return i;
}

bool higher_order (Shape shape)
{
	std::vector<Shape> groups = shape.as_groups();
	return std::any_of(groups.begin(), groups.end(),
		[](Shape& s)
		{
			return 1 < s.n_rank();
		});
}

NElemT index (Shape shape, std::vector<DimT> coord)
{
	uint8_t n = std::min((size_t) shape.n_rank(), coord.size());
	NElemT index = 0;
	for (uint8_t i = 1; i < n; i++)
	{
		index += coord[n - i];
		index *= shape.at(n - i - 1);
	}
	return index + coord[0];
}

std::vector<DimT> coordinate (Shape shape, NElemT idx)
{
	std::vector<DimT> coord;
	DimT xd;
	auto it = shape.begin();
	for (auto et = it + shape.n_rank(); it != et; ++it)
	{
		xd = idx % *it;
		coord.push_back(xd);
		idx = (idx - xd) / *it;
	}
	return coord;
}

#endif
