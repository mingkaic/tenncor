///
/// shape.hpp
/// teq
///
/// Purpose:
/// Define shapes models and coordinate to flattened index mapping
///

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "logs/logs.hpp"

#ifndef TEQ_SHAPE_HPP
#define TEQ_SHAPE_HPP

namespace teq
{

/// Type used for shape rank
using RankT = uint8_t;

/// Type used for shape dimension
#if !defined(SDIM_BYTES) || SDIM_BYTES <= 1
using DimT = uint8_t;
#elif SDIM_BYTES <= 2
using DimT = uint16_t;
#elif SDIM_BYTES <= 4
using DimT = uint32_t;
#else
using DimT = uint64_t;
#endif

/// Type used for flattened index
/// DimT having 8 bits and shape comprising of 8 DimT values means a maximum
/// flattened index of (2 ^ 8) ^ 8 = 2 ^ 64
using NElemT = uint64_t;

/// Number of dimsensions in a shape/coordinate
const RankT rank_cap = 8;

/// Array type used to hold dimension info in Shape
using ShapeT = std::array<DimT,rank_cap>;

/// Type of iterator used to iterate through internal array
using siterator = ShapeT::iterator;

/// Type of constant iterator used to iterate through internal array
using const_siterator = ShapeT::const_iterator;

/// Models an aligned shape using an array of DimT values
/// For each DimT at index i, DimT value is number of elements at dimension i
/// For example, shape={3, 2} can model tensor [[x, y, z], [u, v, w]]
/// (In cartesian coordinate, we treat values along the X-axis as dimension 0)
struct Shape final
{
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

	Shape (Shape&& other) = default;

	Shape& operator = (Shape&& other) = default;

	Shape& operator = (const std::vector<DimT>& dims)
	{
		vector_assign(dims);
		return *this;
	}

	// >>>> ACCESSORS <<<<

	/// Return DimT element at idx for any index in range [0:rank_cap)
	DimT at (RankT idx) const
	{
		if (rank_cap <= idx)
		{
			logs::fatalf("cannot access out of bounds index %d", idx);
		}
		return dims_.at(idx);
	}

	/// Return the total number of elements represented by the shape
	NElemT n_elems (void) const
	{
		auto it = dims_.begin();
		return std::accumulate(it, it + rank_cap, (NElemT) 1,
			std::multiplies<NElemT>());
	}

	/// Return true if this[0:idx) is equal to other[0:idx),
	/// otherwise return false
	bool compatible_before (const Shape& other, RankT idx) const
	{
		auto it = dims_.begin();
		return std::equal(it, it + std::min(idx, rank_cap), other.begin(),
			[](DimT a, DimT b) { return a == 0 || b == 0 || a == b; });
	}

	/// Return true if this[idx:rank_cap) is
	/// equal to other[idx:rank_cap), otherwise return false
	/// Set idx to 0 to compare entire shape
	bool compatible_after (const Shape& other, RankT idx) const
	{
		return idx < rank_cap && std::equal(
			dims_.begin() + idx, dims_.end(), other.begin() + idx,
			[](DimT a, DimT b) { return a == 0 || b == 0 || a == b; });
	}

	/// Return string representation of shape
	std::string to_string (void) const
	{
		return fmts::to_string(dims_.begin(), dims_.end());
	}

	// >>>> INTERNAL CONTROL <<<<

	/// Return begin iterator of internal array
	siterator begin (void)
	{
		return dims_.begin();
	}

	/// Return end iterator of internal array
	siterator end (void)
	{
		return dims_.end();
	}

	/// Return begin constant iterator of internal array
	const_siterator begin (void) const
	{
		return dims_.begin();
	}

	/// Return end constant iterator of internal array
	const_siterator end (void) const
	{
		return dims_.end();
	}

private:
	void vector_assign (const std::vector<DimT>& dims)
	{
		if (std::any_of(dims.begin(), dims.end(),
			[](DimT d)
			{
				return d == 0;
			}))
		{
			logs::fatalf("cannot create shape with vector containing zero: %s",
				fmts::to_string(dims.begin(), dims.end()).c_str());
		}
		RankT rank = std::min((size_t) rank_cap, dims.size());
		auto src = dims.begin();
		auto dest = this->begin();
		std::copy(src, src + rank, dest);
		std::fill(dest + rank, dest + rank_cap, 1);
	}

	/// Array of dimension values
	ShapeT dims_;
};

using ShapesT = std::vector<Shape>;

/// Return list of shape dimensions with trailing ones/zeros trimmed
std::vector<DimT> narrow_shape (const Shape& sign);

}

#endif // TEQ_SHAPE_HPP
