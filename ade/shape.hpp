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

#include "ade/log/log.hpp"

#ifndef ADE_SHAPE_HPP
#define ADE_SHAPE_HPP

namespace ade
{

/// Type used for shape dimension
using DimT = uint8_t;

/// Type used for coordinate dimensions
using CDimT = int16_t;

/// Type used for flattened index
/// DimT having 8 bits and shape comprising of 8 DimT values means a maximum
/// flattened index of (2 ^ 8) ^ 8 = 2 ^ 64
using NElemT = uint64_t;

/// Number of DimT in a shape
const uint8_t rank_cap = 8;

/// Array type used to hold dimension info in Shape
using ShapeT = std::array<DimT,rank_cap>;

/// Array type used to hold dimension info when transforming coordinates
/// Coordinates are allowed to be negative, negative dimensions are counted
/// backward from the corresponding shape dimension
/// For example, given shape=[5], coord=[-1] is the same as coord=[4]
using CoordT = std::array<CDimT,rank_cap>;

/// Models an aligned shape using an array of DimT values
/// For each DimT at index i, DimT value is number of elements at dimension i
/// For example, shape={3, 2} can model tensor [[x, y, z], [u, v, w]]
/// (In cartesian coordinate, we treat values along the X-axis as dimension 0)
struct Shape final
{
	/// Type of iterator used to iterate through internal array
	using iterator = ShapeT::iterator;

	/// Type of constant iterator used to iterate through internal array
	using const_iterator = ShapeT::const_iterator;

	Shape (void)
	{
		std::fill(dims_.begin(), dims_.end(), 1);
	}

	Shape (std::vector<DimT> dims)
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

	/// Return the total number of elements represented by the shape
	NElemT n_elems (void) const
	{
		auto it = dims_.begin();
		return std::accumulate(it, it + rank_cap, 1,
			std::multiplies<NElemT>());
	}

	/// Return true if this->dims_[0:idx) is equal to other.dims_[0:idx),
	/// otherwise return false
	bool compatible_before (const Shape& other, uint8_t idx) const
	{
		auto it = dims_.begin();
		uint8_t cap = rank_cap;
		return std::equal(it, it + std::min(idx, cap), other.dims_.begin());
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
		return ade::to_string(begin(), end());
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
			fatalf("cannot create shape with vector containing zero: %s",
				ade::to_string(dims.begin(), dims.end()).c_str());
		}
		auto dest = dims_.begin();
		uint8_t rank = std::min((size_t) rank_cap, dims.size());
		std::copy(src, src + rank, dest);
		std::fill(dest + rank, dest + rank_cap, 1);
	}

	void move_helper (Shape&& other)
	{
		dims_ = std::move(other.dims_);
		std::fill(other.dims_.begin(), other.dims_.end(), 1);
	}

	/// Array of dimension values
	ShapeT dims_;
};

/// Return the flat index mapped by coord according to shape
///	For example, 2-D tensor has indices in place of value as follows:
/// [[0, 1, ..., n-1], [n, n+1, ..., 2*n-1]]
/// The index follows the equation: index = coord[0]+coord[1]*shape[0]+...
/// Invalid coordinate where the coordinate value is beyond the dimension
/// for any index will report error
NElemT index (Shape shape, CoordT coord);

/// Return the coordinate of a flat index according to shape
/// Coordinate dimensions are 0-based
/// For example [0, 0, ..., 0] <-> 0
CoordT coordinate (Shape shape, NElemT idx);

}

#endif // ADE_SHAPE_HPP
