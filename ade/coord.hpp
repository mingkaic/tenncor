///
///	coord.hpp
///	ade
///
///	Purpose:
///	Define shape/coordinate transformation functions
///

#include <functional>

#include "ade/shape.hpp"

#ifndef ADE_COORD_HPP
#define ADE_COORD_HPP

namespace ade
{

struct iCoordMap
{
	virtual ~iCoordMap (void) = default;

	virtual void forward (Shape::iterator out,
		Shape::const_iterator in) const = 0;

	virtual void backward (Shape::iterator out,
		Shape::const_iterator in) const = 0;

	virtual iCoordMap* reverse (void) const = 0;

	virtual std::string to_string (void) const = 0;
};

using CoordPtrT = std::shared_ptr<iCoordMap>;

extern CoordPtrT identity;

CoordPtrT reduce (uint8_t rank, std::vector<DimT> red);

CoordPtrT extend (uint8_t rank, std::vector<DimT> ext);

CoordPtrT permute (std::vector<uint8_t> order);

CoordPtrT flip (uint8_t dim);

}

#endif // ADE_COORD_HPP
