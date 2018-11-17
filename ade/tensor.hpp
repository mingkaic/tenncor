///
///	tensor.hpp
///	ade
///
///	Purpose:
///	Define interfaces and building blocks for an equation graph
///

#include <cmath>

#include "err/log.hpp"

#include "ade/itensor.hpp"
#include "ade/coord.hpp"

#ifndef ADE_TENSOR_HPP
#define ADE_TENSOR_HPP

namespace ade
{

/// Coordinate mapper and tensor pair
struct MappedTensor final
{
	MappedTensor (CoordPtrT mapper, Tensorptr tensor) :
		mapper_(mapper), tensor_(tensor) {}

	/// Return shape of tensor filtered through coordinate mapper
	Shape shape (void) const
	{
		const Shape& shape = tensor_->shape();
		CoordT out;
		CoordT in;
		std::copy(shape.begin(), shape.end(), in.begin());
		mapper_->forward(out.begin(), in.begin());
		std::vector<DimT> slist(rank_cap);
		std::transform(out.begin(), out.end(), slist.begin(),
			[](CDimT cd) -> DimT
			{
				if (cd < 0)
				{
					cd = -cd - 1;
				}
				return std::round(cd);
			});
		return Shape(slist);
	}

	/// Coordinate mapper
	CoordPtrT mapper_;

	/// Tensor reference
	Tensorptr tensor_;
};

/// Leaf of the graph commonly representing the variable in an equation
struct Tensor : public iTensor
{
	virtual ~Tensor (void) = default;

	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Return pointer to internal data
	virtual void* data (void) = 0;

	/// Return const pointer to internal data
	virtual const void* data (void) const = 0;

	/// Return data type encoding
	virtual size_t type_code (void) const = 0;
};

}

#endif // ADE_TENSOR_HPP
