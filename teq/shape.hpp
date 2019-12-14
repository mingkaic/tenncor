///
/// shape.hpp
/// teq
///
/// Purpose:
/// Define shapes models and coordinate to flattened index mapping
///

#include "teq/ishape.hpp"

#ifndef TEQ_SHAPE_HPP
#define TEQ_SHAPE_HPP

namespace teq
{

/// Type used for coordinate dimensions
using CDimT = double;

/// Array type used to hold dimension info when transforming coordinates
/// Coordinates are allowed to be negative, negative dimensions are counted
/// backward from the corresponding shape dimension
/// For example, given shape=[5], coord=[-1] is the same as coord=[4]
using CoordT = std::array<CDimT,rank_cap>;

/// Models an aligned shape using an array of DimT values
/// For each DimT at index i, DimT value is number of elements at dimension i
/// For example, shape={3, 2} can model tensor [[x, y, z], [u, v, w]]
/// (In cartesian coordinate, we treat values along the X-axis as dimension 0)
struct Shape final : public ShapeSignature
{
	Shape (void)
	{
		std::fill(begin(), end(), 1);
	}

	Shape (std::vector<DimT> dims) :
		ShapeSignature(dims)
	{
		validate_shape();
	}

	Shape (const ShapeSignature& sign) :
		ShapeSignature(sign)
	{
		validate_shape();
	}

	Shape& operator = (const ShapeSignature& other)
	{
		if (&other != this)
		{
			ShapeSignature::operator = (other);
			validate_shape();
		}
		return *this;
	}

	Shape (ShapeSignature&& other) :
		ShapeSignature(std::move(other))
	{
		validate_shape();
	}

	Shape& operator = (ShapeSignature&& other)
	{
		if (&other != this)
		{
			ShapeSignature::operator = (std::move(other));
			validate_shape();
		}
		return *this;
	}

	Shape& operator = (const std::vector<DimT>& dims)
	{
		this->vector_assign(dims);
		validate_shape();
		return *this;
	}

	// >>>> ACCESSORS <<<<

	/// Return the total number of elements represented by the shape
	NElemT n_elems (void) const
	{
		auto it = begin();
		return std::accumulate(it, it + rank_cap, (NElemT) 1,
			std::multiplies<NElemT>());
	}

private:
	void validate_shape (void) const
	{
		if (std::any_of(begin(), end(),
			[](DimT d)
			{
				return d == 0;
			}))
		{
			logs::fatalf("cannot create shape with vector containing zero: %s",
				fmts::to_string(begin(), end()).c_str());
		}
	}
};

/// Return the flat index mapped by coord according to shape
/// For example, 2-D tensor has indices in place of value as follows:
/// [[0, 1, ..., n-1], [n, n+1, ..., 2*n-1]]
/// The index follows the equation: index = coord[0]+coord[1]*shape[0]+...
/// Invalid coordinate where the coordinate value is beyond the dimension
/// for any index will report error
NElemT index (const Shape& shape, CoordT coord);

/// Return the coordinate of a flat index according to shape
/// Coordinate dimensions are 0-based
/// For example [0, 0, ..., 0] <-> 0
CoordT coordinate (const Shape& shape, NElemT idx);

/// Return list of shape dimensions with trailing ones/zeros trimmed
std::vector<DimT> narrow_shape (const ShapeSignature& sign);

bool is_ambiguous (const ShapeSignature& sign);

}

#endif // TEQ_SHAPE_HPP
