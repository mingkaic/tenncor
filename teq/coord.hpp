///
/// coord.hpp
/// teq
///
/// Purpose:
/// Define shape/coordinate transformation functions
///

#include <functional>

#include "teq/matops.hpp"

#ifndef TEQ_COORD_HPP
#define TEQ_COORD_HPP

namespace teq
{

/// Interface for transforming coordinates and reversing the coordinate
struct iCoordMap
{
	virtual ~iCoordMap (void) = default;

	/// Return matmul(this, rhs)
	virtual iCoordMap* connect (const iCoordMap& rhs) const = 0;

	/// Forward transform coordinates
	virtual void forward (CoordT::iterator out,
		CoordT::const_iterator in) const = 0;

	/// Return coordinate transformation with its forward and backward
	/// transformations reversed
	virtual iCoordMap* reverse (void) const = 0;

	/// Return string representation of coordinate transformer
	virtual std::string to_string (void) const = 0;

	/// Access the forward matrix representation of transformer as a param to input
	/// callback function cb
	virtual void access (std::function<void(const MatrixT&)> cb) const = 0;

	/// Return true if this instance maps coordinates/shapes bijectively
	virtual bool is_bijective (void) const = 0;
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
	}

	/// Implementation of iCoordMap
	iCoordMap* connect (const iCoordMap& rhs) const override
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
	void forward (CoordT::iterator out,
		CoordT::const_iterator in) const override;

	/// Implementation of iCoordMap
	iCoordMap* reverse (void) const override
	{
		return new CoordMap([this](MatrixT m)
		{
			inverse(m, this->fwd_);
		});
	}

	/// Implementation of iCoordMap
	std::string to_string (void) const override
	{
		return teq::to_string(fwd_);
	}

	/// Implementation of iCoordMap
	void access (std::function<void(const MatrixT&)> cb) const override
	{
		cb(fwd_);
	}

	/// Implementation of iCoordMap
	bool is_bijective (void) const override
	{
		return (int) determinant(fwd_) != 0;
	}

private:
	/// Forward transformation matrix
	MatrixT fwd_;
};

/// Type of iCoordMap smartpointer
using CoordptrT = std::shared_ptr<iCoordMap>;

/// Identity matrix instance
extern CoordptrT identity;

/// Checks if the coord mapper is an identity mapper
bool is_identity (iCoordMap* coorder);

/// Return coordinate mapper dividing dimensions after rank
/// by values in red vector
/// For example, given coordinate [2, 2, 6, 6], rank=2, and red=[3, 3],
/// mapper forward transforms to coordinate [2, 2, 2, 2]
CoordptrT reduce (RankT rank, std::vector<DimT> red);

/// Return coordinate mapper multiplying dimensions after rank
/// by values in ext vector
/// For example, given coordinate [6, 6, 2, 2], rank=2, and ext=[3, 3],
/// mapper forward transforms to coordinate [6, 6, 6, 6]
CoordptrT extend (RankT rank, std::vector<DimT> ext);

/// Return coordinate mapper permuting coordinate according to input order
/// Order is a vector of indices of the dimensions to appear in order
/// Indices not referenced by order but less than rank_cap will be appended
/// by numerical order
/// For example, given coordinate [1, 2, 3, 4], order=[1, 3],
/// mapper forward transforms to coordinate [2, 4, 1, 3]
/// Returned coordinate mapper will be a CoordMap instance, so inversibility
/// requires order indices be unique, otherwise throw fatal error
CoordptrT permute (std::vector<RankT> order);

/// Return coordinate mapper flipping coordinate value at specified dimension
/// Flipped dimension with original value x is represented as -x-1
/// (see CoordT definition)
CoordptrT flip (RankT dim);

}

#endif // TEQ_COORD_HPP