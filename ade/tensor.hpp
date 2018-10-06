///
///	tensor.hpp
///	ade
///
///	Purpose:
///	Define interfaces and building blocks for an equation graph
///

#include <memory>

#include "util/error.hpp"

#include "ade/shape.hpp"

#ifndef ADE_TENSOR_HPP
#define ADE_TENSOR_HPP

namespace ade
{

struct Tensorptr;

/// Interface for ensuring derivation functionality, and shape encapsulation
struct iTensor
{
	virtual ~iTensor (void) = default;

	/// Return the shape held by this tensor
	virtual const Shape& shape (void) const = 0;

	/// Return the root of the partial derivative with respect to input wrt
	virtual Tensorptr gradient (Tensorptr& wrt) const = 0;

	/// Return the string representation of the tensor
	virtual std::string to_string (void) const = 0;
};

/// Smart pointer to iTensor ensuring non-null references
struct Tensorptr
{
	Tensorptr (iTensor& tens) :
		ptr_(&tens) {}

	Tensorptr (iTensor* tens) :
		ptr_(tens)
	{
		if (nullptr == tens)
		{
			util::handle_error("init nodeptr with nullptr");
		}
	}

	virtual ~Tensorptr (void) = default;

	iTensor* operator -> (void)
	{
		return ptr_.get();
	}

	const iTensor* operator -> (void) const
	{
		return ptr_.get();
	}

	/// Return the raw pointer
	iTensor* get (void) const
	{
		return ptr_.get();
	}

	/// Return the weakptr reference
	std::weak_ptr<iTensor> ref (void) const
	{
		return ptr_;
	}

protected:
	/// Strong reference to iTensor
	std::shared_ptr<iTensor> ptr_;
};

/// Return a Tensor::SYMBOLIC_ONE reshaped to input shape
Tensorptr constant_one (std::vector<DimT> shape);

/// Return a Tensor::SYMBOLIC_ZERO reshaped to input shape
Tensorptr constant_zero (std::vector<DimT> shape);

/// Leaf of the graph commonly representing the variable in an equation
struct Tensor final : public iTensor
{
	/// Represent a scalar containing value one
	static Tensorptr SYMBOLIC_ONE;

	/// Represent a scalar containing value zero
	static Tensorptr SYMBOLIC_ZERO;

	/// Return a Tensor with input shape
	static Tensorptr get (Shape shape)
	{
		return new Tensor(shape);
	}

	/// Implementation of iTensor
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	Tensorptr gradient (Tensorptr& wrt) const override
	{
		std::vector<DimT> shape = wrt->shape().as_list();
		if (this == wrt.get())
		{
			return constant_one(shape);
		}
		return constant_zero(shape);
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
