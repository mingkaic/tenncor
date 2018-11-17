///
///	coord.hpp
///	ade
///
///	Purpose:
///	Define shape/coordinate transformation functions
///

#include <cstring>
#include <functional>

#include "ade/shape.hpp"
#include "ade/matops.hpp"

#ifndef ADE_COORD_HPP
#define ADE_COORD_HPP

namespace ade
{

/// Interface for transforming coordinates and reversing the coordinate
struct iCoordMap
{
	virtual ~iCoordMap (void) = default;

	/// Forward transform coordinates
	virtual void forward (CoordT::iterator out,
		CoordT::const_iterator in) const = 0;

	/// Return matmul(this, rhs)
	virtual iCoordMap* forward (const iCoordMap& rhs) const = 0;

	/// Reverse transform coordinates
	virtual void backward (CoordT::iterator out,
		CoordT::const_iterator in) const = 0;

	/// Return coordinate transformation with its forward and backward
	/// transformations reversed
	virtual iCoordMap* reverse (void) const = 0;

	/// Return string representation of coordinate transformer
	virtual std::string to_string (void) const = 0;

	/// Access the matrix representation of transformer as a param to input
	/// callback function cb
	virtual void access (std::function<void(const MatrixT&)> cb) const = 0;
};

/// Coordinate transformation implementation using homogeneous matrices
/// The transformation matrix must be inversible otherwise fatal on creation
struct CoordMap final : public iCoordMap
{
	CoordMap (std::function<void(MatrixT)> init)
	{
		std::memset(fwd_, 0, mat_size);
		fwd_[rank_cap][rank_cap] = 1;
		init(fwd_);
		inverse(bwd_, fwd_);
	}

	/// Implementation of iCoordMap
	void forward (CoordT::iterator out,
		CoordT::const_iterator in) const override;

	/// Implementation of iCoordMap
	iCoordMap* forward (const iCoordMap& rhs) const override
	{
		return new CoordMap([&](MatrixT out)
		{
			rhs.access([&](const MatrixT& in)
			{
				matmul(out, fwd_, in);
			});
		});
	}

	/// Implementation of iCoordMap
	void backward (CoordT::iterator out,
		CoordT::const_iterator in) const override;

	/// Implementation of iCoordMap
	iCoordMap* reverse (void) const override
	{
		return new CoordMap(bwd_, fwd_);
	}

	/// Implementation of iCoordMap
	std::string to_string (void) const override
	{
		return ade::to_string(fwd_);
	}

	/// Implementation of iCoordMap
	void access (std::function<void(const MatrixT&)> cb) const override
	{
		cb(fwd_);
	}

private:
	CoordMap (const MatrixT fwd, const MatrixT bwd)
	{
		std::memcpy(fwd_, fwd, mat_size);
		std::memcpy(bwd_, bwd, mat_size);
	}

	/// Forward transformation matrix
	MatrixT fwd_;

	/// Inverse of the forward transformation matrix
	MatrixT bwd_;
};

/// Type of iCoordMap smartpointer
using CoordPtrT = std::shared_ptr<iCoordMap>;

/// Identity matrix instance
extern CoordPtrT identity;

/// Return coordinate mapper dividing dimensions after rank
/// by values in red vector
/// For example, given coordinate [2, 2, 6, 6], rank=2, and red=[3, 3],
/// mapper forward transforms to coordinate [2, 2, 2, 2]
CoordPtrT reduce (uint8_t rank, std::vector<DimT> red);

/// Return coordinate mapper multiplying dimensions after rank
/// by values in ext vector
/// For example, given coordinate [6, 6, 2, 2], rank=2, and ext=[3, 3],
/// mapper forward transforms to coordinate [6, 6, 6, 6]
CoordPtrT extend (uint8_t rank, std::vector<DimT> ext);

/// Return coordinate mapper permuting coordinate according to input order
/// Order is a vector of indices of the dimensions to appear in order
/// Indices not referenced by order but less than rank_cap will be appended
/// by numerical order
/// For example, given coordinate [1, 2, 3, 4], order=[1, 3],
/// mapper forward transforms to coordinate [2, 4, 1, 3]
/// Returned coordinate mapper will be a CoordMap instance, so inversibility
/// requires order indices be unique, otherwise throw fatal error
CoordPtrT permute (std::vector<uint8_t> order);

/// Return coordinate mapper flipping coordinate value at specified dimension
/// Flipped dimension with original value x is represented as -x-1
/// (see CoordT definition)
CoordPtrT flip (uint8_t dim);

}

#endif // ADE_COORD_HPP
