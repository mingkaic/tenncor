#include "internal/teq/shape.hpp"

#ifndef TEQ_COORD_HPP
#define TEQ_COORD_HPP

namespace teq
{

/// Type used for coordinate dimensions
using CDimT = double;

/// Array type used to hold dimension info when transforming coordinates
/// Coordinates are allowed to be negative, negative dimensions are counted
/// backward from the corresponding shape dimension
/// For example, given shape=[5], coord=[-1] is the same as coord=[4]
using CoordT = std::array<CDimT,rank_cap>;

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

}

#endif // TEQ_COORD_HPP
