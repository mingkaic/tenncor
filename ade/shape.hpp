#include <algorithm>
#include <array>
#include <iterator>
#include <numeric>
#include <sstream>
#include <vector>

#include "util/error.hpp"

#ifndef ADE_SHAPE_HPP
#define ADE_SHAPE_HPP

namespace ade
{

using DimT = uint8_t;
using NElemT = uint64_t; // reliant on rank_cap
const uint8_t rank_cap = 8;

// aligned shape representation
struct Shape
{
	using ShapeIterator = std::array<DimT,rank_cap>::iterator;

	Shape (void) : rank_(0)
	{
		std::fill(dims_.begin(), dims_.end(), 1);
	}

	Shape (std::vector<DimT> dims) : rank_(rank_cap)
	{
		vector_assign(dims);
	}

	Shape (const Shape& other) = default;

	Shape& operator = (const Shape& other) = default;

	Shape& operator = (const std::vector<DimT>& dims)
	{
		vector_assign(dims);
		return *this;
	}

	Shape (Shape&& other)
	{
		move_helper(std::move(other));
	}

	Shape& operator = (Shape&& other)
	{
		if (this != &other)
		{
			move_helper(std::move(other));
		}
		return *this;
	}


	// >>>> ACCESSORS <<<<

	DimT at (uint8_t idx) const
	{
		if (idx >= rank_cap)
		{
			throw std::out_of_range(
				"accessing dimension out of allocated rank cap");
		}
		return dims_.at(idx);
	}

	uint8_t n_rank (void) const
	{
		return rank_;
	}

	NElemT n_elems (void) const
	{
		auto it = dims_.begin();
		return std::accumulate(it, it + rank_, 1,
			std::multiplies<NElemT>());
	}

	// vectorize dim up to rank
	std::vector<DimT> as_list (void) const
	{
		auto it = dims_.begin();
		return std::vector<DimT>(it, it + rank_);
	}

	bool compatible_before (const Shape& other, uint8_t idx) const
	{
		auto it = dims_.begin();
		uint8_t cap = rank_cap;
		return std::equal(it,
			it + std::min(idx, cap),
			other.dims_.begin());
	}

	//! check if this is compatible
	//! with another shape after a certain rank
	bool compatible_after (const Shape& other, uint8_t idx) const
	{
		bool compatible = false;
		if (idx < rank_cap)
		{
			compatible = std::equal(dims_.begin() + idx,
				dims_.end(), other.dims_.begin() + idx);
		}
		return compatible;
	}

	std::string to_string (void) const
	{
		std::stringstream ss;
		util::to_stream(ss, as_list());
		return ss.str();
	}

	// >>>> ITERATORS <<<<

	ShapeIterator begin (void)
	{
		return dims_.begin();
	}

	ShapeIterator end (void)
	{
		return dims_.end();
	}

private:
	void vector_assign (const std::vector<DimT>& dims)
	{
		auto src = dims.begin();
		if (std::any_of(src, dims.end(),
			[](DimT d)
			{
				return d == 0;
			}))
		{
			util::handle_error("shape assignment with zero vector",
				util::ErrArg<std::vector<DimT>>{"vec", dims});
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

	void move_helper (Shape&& other)
	{
		dims_ = std::move(other.dims_);
		rank_ = std::move(other.rank_);

		auto ot = other.dims_.begin();
		std::fill(ot, ot + rank_, 1);
		other.rank_ = 0;
	}

	std::array<DimT,rank_cap> dims_;

	uint8_t rank_;
};

//! obtain the flat vector index from cartesian coordinates
//! (e.g.: 2-D [x, y] has flat index = y * dimensions_[0] + x)
NElemT index (Shape shape, std::vector<DimT> coord);

//! obtain cartesian coordinates given a flat vector index
std::vector<DimT> coordinate (Shape shape, NElemT idx);

}

#endif /* ADE_SHAPE_HPP */
