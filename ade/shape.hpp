/*!
 *
 *  shape.hpp
 *  ade
 *
 *  Purpose:
 *  shapes are arrays providing shape and coordinate utility functions
 *
 */

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

/*! limit on the rank of the shape */
const uint8_t rank_cap = 8;

// aligned shape representation
struct Shape
{
	using ShapeIterator = std::array<DimT,rank_cap>::iterator;

	using ConstShapeIterators = std::array<DimT,rank_cap>::const_iterator;

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

	/*! get element at idx. idx can be greater than n_rank, but must be less than rank_cap */
	DimT at (uint8_t idx) const
	{
		if (idx >= rank_cap)
		{
			throw std::out_of_range(
				"accessing dimension out of allocated rank cap");
		}
		return dims_.at(idx);
	}

	/*! rank size containing meaningful dimensions */
	uint8_t n_rank (void) const
	{
		return rank_;
	}

	/*! total number of elements represented by shape */
	NElemT n_elems (void) const
	{
		auto it = dims_.begin();
		return std::accumulate(it, it + rank_, 1,
			std::multiplies<NElemT>());
	}

	/*! get vector of meaningful dimensions */
	std::vector<DimT> as_list (void) const
	{
		auto it = dims_.begin();
		return std::vector<DimT>(it, it + rank_);
	}

	/*! check if sub-shape up to idx is equal to another sub-shape up to idx */
	bool compatible_before (const Shape& other, uint8_t idx) const
	{
		auto it = dims_.begin();
		uint8_t cap = rank_cap;
		return std::equal(it,
			it + std::min(idx, cap),
			other.dims_.begin());
	}

	/*! check if sub-shape after idx is equal to another sub-shape after idx,
	 *	set idx to 0 to compare entire shape */
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

	/*! represent the shape as string (for debugging purposes) */
	std::string to_string (void) const
	{
		std::stringstream ss;
		util::to_stream(ss, as_list());
		return ss.str();
	}

	// >>>> ITERATORS <<<<

	/*! get beginning iterator for internal dimension array */
	ShapeIterator begin (void)
	{
		return dims_.begin();
	}

	/*! get end iterator for internal dimension array
	 *	(this may extend beyond range of n_rank) */
	ShapeIterator end (void)
	{
		return dims_.end();
	}

	/*! get constant beginning iterator for internal dimension array */
	ConstShapeIterators begin (void) const
	{
		return dims_.begin();
	}

	/*! get constant end iterator for internal dimension array */
	ConstShapeIterators end (void) const
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

	/*! internal dimension array */
	std::array<DimT,rank_cap> dims_;

	/*! number of meaningful dimensions */
	uint8_t rank_;
};

/*! obtain the flat vector index from cartesian coordinates
 *	(e.g.: 2-D [x, y] has flat index = y * dimensions_[0] + x) */
NElemT index (Shape shape, std::vector<DimT> coord);

/*! obtain cartesian coordinates given a flat vector index */
std::vector<DimT> coordinate (Shape shape, NElemT idx);

}

#endif /* ADE_SHAPE_HPP */
