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

struct MappedTensor final
{
	MappedTensor (CoordPtrT mapper, Tensorptr tensor) :
		mapper_(mapper), tensor_(tensor) {}

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

	CoordPtrT mapper_;
	Tensorptr tensor_;
};

/// Interface for holding data when passing up the tensor graph
struct iData
{
	virtual ~iData (void) = default;

	virtual char* get (void) = 0;

	virtual const char* get (void) const = 0;

	virtual size_t type_code (void) const = 0;
};

/// Leaf of the graph commonly representing the variable in an equation
struct Tensor : public iTensor
{
	/// Implementation of iTensor
	void accept (iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Implementation of iTensor
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

	virtual iData& data (void) = 0;

protected:
	Tensor (Shape shape) : shape_(shape) {}

	/// Shape info of the tensor instance
	Shape shape_;
};

}

#endif // ADE_TENSOR_HPP
