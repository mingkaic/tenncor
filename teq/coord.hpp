///
/// coord.hpp (todo: rename to convert)
/// teq
///
/// Purpose:
/// Define shape/coordinate transformation functions
///

#include <functional>
#include <set>

#include "teq/matops.hpp"

#ifndef TEQ_COORD_HPP
#define TEQ_COORD_HPP

namespace teq
{

/// Interface for transforming coordinates and reversing the coordinate
struct iConvert
{
	virtual ~iConvert (void) = default;

	/// Return string representation of converter
	virtual std::string to_string (void) const = 0;

	/// Access the forward matrix representation of transformer as a param to input
	/// callback function cb
	virtual void access (std::function<void(const MatrixT&)> cb) const = 0;
};

/// Type of iConvert smartpointer
using CvrtptrT = std::shared_ptr<iConvert>;

using MatInitF = std::function<void(MatrixT&)>;

/// Coordinate transformation implementation using homogeneous matrices
/// The transformation matrix must be inversible otherwise fatal on creation
struct ShapeMap final
{
	ShapeMap (MatInitF init)
	{
		std::fill(fwd_[0], fwd_[0] + mat_size, 0);
		fwd_[rank_cap][rank_cap] = 1;
		init(fwd_);
	}

	/// Return converted shape
	Shape convert (const Shape& shape) const;

	/// Return string representation of mapper
	std::string to_string (void) const
	{
		return ::teq::to_string(fwd_);
	}

	/// Access the forward matrix representation of transformer as a param to input
	/// callback function cb
	void access (std::function<void(const MatrixT&)> cb) const
	{
		cb(fwd_);
	}

private:
	/// Forward transformation matrix
	MatrixT fwd_;
};

using ShaperT = std::shared_ptr<ShapeMap>;

/// Identity matrix instance
extern ShaperT identity;

/// Checks if the coord mapper is an identity mapper
bool is_identity (ShapeMap* shaper);

ShaperT reduce (std::set<RankT> rdims);

ShaperT extend (teq::CoordT bcast);

/// Return coordinate mapper permuting coordinate according to input order
/// Order is a vector of indices of the dimensions to appear in order
/// Indices not referenced by order but less than rank_cap will be appended
/// by numerical order
/// For example, given coordinate [1, 2, 3, 4], order=[1, 3],
/// mapper forward transforms to coordinate [2, 4, 1, 3]
/// Returned coordinate mapper will be a CoordMap instance, so inversibility
/// requires order indices be unique, otherwise throw fatal error
ShaperT permute (std::array<RankT,rank_cap> order);


// todo: test above reduce/extend then remove below funcs
/// Return coordinate mapper dividing dimensions after rank
/// by values in red vector
/// For example, given coordinate [2, 2, 6, 6], rank=2, and red=[3, 3],
/// mapper forward transforms to coordinate [2, 2, 2, 2]
ShaperT reduce (RankT rank, std::vector<DimT> red);

/// Return coordinate mapper multiplying dimensions after rank
/// by values in ext vector
/// For example, given coordinate [6, 6, 2, 2], rank=2, and ext=[3, 3],
/// mapper forward transforms to coordinate [6, 6, 6, 6]
ShaperT extend (RankT rank, std::vector<DimT> ext);

}

#endif // TEQ_COORD_HPP
