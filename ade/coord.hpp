///
///	coord.hpp
///	ade
///
///	Purpose:
///	Define shape/coordinate transformation functions
///

#include <functional>

#include "ade/shape.hpp"
#include "ade/matops.hpp"

#ifndef ADE_COORD_HPP
#define ADE_COORD_HPP

namespace ade
{

struct iCoordMap
{
	virtual ~iCoordMap (void) = default;

	virtual void forward (CoordT::iterator out,
		CoordT::const_iterator in) const = 0;

	virtual void backward (CoordT::iterator out,
		CoordT::const_iterator in) const = 0;

	virtual iCoordMap* reverse (void) const = 0;

	virtual std::string to_string (void) const = 0;

	virtual void access (std::function<void(const MatrixT&)> acc) const = 0;
};

struct CoordMap final : public iCoordMap
{
	CoordMap (std::function<void(MatrixT)> init)
	{
		std::memset(fwd_, 0, mat_size);
		fwd_[rank_cap][rank_cap] = 1;
		init(fwd_);
		inverse(bwd_, fwd_);
	}

	void forward (CoordT::iterator out,
		CoordT::const_iterator in) const override;

	void backward (CoordT::iterator out,
		CoordT::const_iterator in) const override;

	iCoordMap* reverse (void) const override
	{
		return new CoordMap(bwd_, fwd_);
	}

	std::string to_string (void) const override
	{
		return ade::to_string(fwd_);
	}

	void access (std::function<void(const MatrixT&)> acc) const override
	{
		acc(fwd_);
	}

private:
	CoordMap (const MatrixT fwd, const MatrixT bwd)
	{
		std::memcpy(fwd_, fwd, mat_size);
		std::memcpy(bwd_, bwd, mat_size);
	}

	MatrixT fwd_;
	MatrixT bwd_;
};

using CoordPtrT = std::shared_ptr<iCoordMap>;

extern CoordPtrT identity;

Shape map_shape (CoordPtrT& mapper, const Shape& shape);

CoordPtrT reduce (uint8_t rank, std::vector<DimT> red);

CoordPtrT extend (uint8_t rank, std::vector<DimT> ext);

CoordPtrT permute (std::vector<uint8_t> order);

CoordPtrT flip (uint8_t dim);

}

#endif // ADE_COORD_HPP
