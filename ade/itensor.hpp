///
///	tensor.hpp
///	ade
///
///	Purpose:
///	Define interfaces and building blocks for an equation graph
///

#include <memory>

#include "ade/shape.hpp"

#ifndef ADE_INTERFACE_HPP
#define ADE_INTERFACE_HPP

namespace ade
{

struct Tensorptr;

struct Tensor;

struct iFunctor;

/// Interface to travel through graph, treating Tensor and iFunctor differently
struct iTraveler
{
	virtual ~iTraveler (void) = default;

	/// Visit leaf node
	virtual void visit (Tensor* leaf) = 0;

	/// Visit functor node
	virtual void visit (iFunctor* func) = 0;
};

/// Interface of traversible and differentiable nodes with shape information
struct iTensor
{
	virtual ~iTensor (void) = default;

	/// Obtain concrete information on either leaf or functor implementations
	virtual void accept (iTraveler& visiter) = 0;

	/// Return the shape held by this tensor
	virtual const Shape& shape (void) const = 0;

	/// Return the partial derivative of this with respect to input wrt
	virtual Tensorptr gradient (const iTensor* wrt) = 0;

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
			fatal("cannot create nodeptr with nullptr");
		}
	}

	Tensorptr (std::shared_ptr<iTensor> tens) :
		ptr_(tens)
	{
		if (nullptr == tens)
		{
			fatal("cannot create nodeptr with nullptr");
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

}

#endif // ADE_INTERFACE_HPP