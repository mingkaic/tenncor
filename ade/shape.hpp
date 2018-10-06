///
///	shape.hpp
///	ade
///
///	Purpose:
///	Define shapes models and coordinate to flattened index mapping
///

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

/// Type used for each dimension
using DimT = uint8_t;

/// Type used for flattened index
/// DimT having 8 bits and shape comprising of 8 DimT values means a maximum
/// flattened index of (2 ^ 8) ^ 8 = 2 ^ 64
using NElemT = uint64_t;

/// Number of DimT in a shape
/// The logical rank of a shape can be less than rank_cap
const uint8_t rank_cap = 8;

/// Models an aligned shape using an array of DimT values
/// For each DimT at index i, DimT value is number of elements at dimension i
/// For example, shape={3, 2} can model tensor [[x, y, z], [u, v, w]]
/// (In cartesian coordinate, we treat values along the X-axis as dimension 0)
/// Shapes will always hold a fixed number of values, but may vary in rank
/// Rank is the number of meaningful dimensions, values beyond the rank are 1
/// For example, a 2 rank shape of {3, 2} is internally {3, 2, 1, 1, 1, ...}
/// This choice is intended to simplify shape manipulation functions
struct Shape final
{
	/// Type of iterator used to iterate through internal array
	using iterator = std::array<DimT,rank_cap>::iterator;

	/// Type of constant iterator used to iterate through internal array
	using const_iterator = std::array<DimT,rank_cap>::const_iterator;

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

	/// Return DimT element at idx for any index in range [0:rank_cap)
	DimT at (uint8_t idx) const
	{
		if (idx >= rank_cap)
		{
			throw std::out_of_range(
				"accessing dimension out of allocated rank cap");
		}
		return dims_.at(idx);
	}

	/// Return the rank (number of guaranteed meaningful dimensions)
	uint8_t n_rank (void) const
	{
		return rank_;
	}

	/// Return the total number of elements represented by the shape
	NElemT n_elems (void) const
	{
		auto it = dims_.begin();
		return std::accumulate(it, it + rank_, 1,
			std::multiplies<NElemT>());
	}

	/// Return shape as a vector of only the meaningful dimensions
	std::vector<DimT> as_list (void) const
	{
		auto it = dims_.begin();
		return std::vector<DimT>(it, it + rank_);
	}

	/// Return true if this->dims_[0:idx) is equal to other.dims_[0:idx),
	/// otherwise return false
	bool compatible_before (const Shape& other, uint8_t idx) const
	{
		auto it = dims_.begin();
		uint8_t cap = rank_cap;
		return std::equal(it,
			it + std::min(idx, cap),
			other.dims_.begin());
	}

	/// Return true if this->dims_[idx:rank_cap) is
	/// equal to other.dims_[idx:rank_cap), otherwise return false
	///	Set idx to 0 to compare entire shape
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

	/// Return string representation of shape
	std::string to_string (void) const
	{
		std::stringstream ss;
		util::to_stream(ss, as_list());
		return ss.str();
	}

	// >>>> INTERNAL CONTROL <<<<

	/// Return begin iterator of internal array
	iterator begin (void)
	{
		return dims_.begin();
	}

	/// Return end iterator of internal array
	iterator end (void)
	{
		return dims_.end();
	}

	/// Return begin constant iterator of internal array
	const_iterator begin (void) const
	{
		return dims_.begin();
	}

	/// Return end constant iterator of internal array
	const_iterator end (void) const
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

	/// Array of dimension values, we hide these to prevent assignment
	/// and ensure rank_ and dims_ relationship
	std::array<DimT,rank_cap> dims_;

	/// Guaranteed number of meaningful dimensions
	uint8_t rank_;
};

/// Return the flat index mapped by coord according to shape
///	For example, 2-D tensor has indices in place of value as follows:
/// [[0, 1, ..., n-1], [n, n+1, ..., 2*n-1]]
/// The index follows the equation: index = coord[0]+coord[1]*shape[0]+...
/// Invalid coordinate where the coordinate value is beyond the dimension
/// for any index will report error
NElemT index (Shape shape, std::vector<DimT> coord);

/// Return the coordinate of a flat index according to shape
std::vector<DimT> coordinate (Shape shape, NElemT idx);

}

#endif // ADE_SHAPE_HPP
