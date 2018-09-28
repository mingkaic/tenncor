/*!
 *
 *  tensor.hpp
 *  ade
 *
 *  Purpose:
 *  define building blocks for an equation tree
 *
 */

#include <memory>

#include "util/error.hpp"

#include "ade/shape.hpp"

#ifndef ADE_TENSOR_HPP
#define ADE_TENSOR_HPP

namespace ade
{

struct Tensorptr;

/*! Tensor interface for ensuring derivation functionality */
struct iTensor
{
	virtual ~iTensor (void) = default;

	/*! get internal shape */
	virtual const Shape& shape (void) const = 0;

	/*! build the gradient subtree with respect to wrt node */
	virtual Tensorptr gradient (Tensorptr& wrt) const = 0;

	/*! represent tensor information as string (for debug purposes) */
	virtual std::string to_string (void) const = 0;
};

/*! Internal smart pointer wrapper to Tensors guaranteeing non-null pointers */
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

	/*! get raw pointer */
	iTensor* get (void) const
	{
		return ptr_.get();
	}

	/*! get weakptr reference */
	std::weak_ptr<iTensor> ref (void) const
	{
		return ptr_;
	}

protected:
	/*! smartpointer to iTensor */
	std::shared_ptr<iTensor> ptr_;
};

/*! Reshaping Tensor::SYMBOLIC_ONE to input shape */
Tensorptr constant_one (std::vector<DimT> shape);

/*! Reshaping Tensor::SYMBOLIC_ZERO to input shape */
Tensorptr constant_zero (std::vector<DimT> shape);

/*! Tensor implementation representing leaf node in operation graph */
struct Tensor final : public iTensor
{
	/*! representation for a scalar containing value one */
	static Tensorptr SYMBOLIC_ONE;

	/*! representation for a scalar containing value zero */
	static Tensorptr SYMBOLIC_ZERO;

	/*! build a tensor of input shape */
	static Tensorptr get (Shape shape)
	{
		return new Tensor(shape);
	}

	/*! implementation of iTensor  */
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/*! implementation of iTensor  */
	Tensorptr gradient (Tensorptr& wrt) const override
	{
		std::vector<DimT> shape = wrt->shape().as_list();
		if (this == wrt.get())
		{
			return constant_one(shape);
		}
		return constant_zero(shape);
	}

	/*! implementation of iTensor  */
	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

private:
	Tensor (Shape shape) : shape_(shape) {}

	/*! internal shape  */
	Shape shape_;
};

}

#endif /* ADE_TENSOR_HPP */
