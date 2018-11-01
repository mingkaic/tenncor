///
///	tensor.hpp
///	ade
///
///	Purpose:
///	Define interfaces and building blocks for an equation graph
///

#include <memory>

#include "ade/log/log.hpp"

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
				return cd;
			});
		return Shape(slist);
	}

	CoordPtrT mapper_;
	Tensorptr tensor_;
};

/// Return a Tensor::SYMBOLIC_ONE extended to input shape
Tensorptr shaped_one (Shape shape);

/// Return a Tensor::SYMBOLIC_ZERO extended to input shape
Tensorptr shaped_zero (Shape shape);

/// Leaf of the graph commonly representing the variable in an equation
struct Tensor final : public iTensor
{
	/// Represent a scalar containing value one
	static Tensorptr SYMBOLIC_ONE;

	/// Represent a scalar containing value zero
	static Tensorptr SYMBOLIC_ZERO;

	/// Return a Tensor with input shape
	static Tensor* get (Shape shape)
	{
		return new Tensor(shape);
	}

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
	Tensorptr gradient (const iTensor* wrt) override
	{
		if (this == wrt)
		{
			return shaped_one(shape_);
		}
		return shaped_zero(wrt->shape());
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

private:
	Tensor (Shape shape) : shape_(shape) {}

	/// Shape info of the tensor instance
	Shape shape_;
};

}

#endif // ADE_TENSOR_HPP
