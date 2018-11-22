///
///	itensor.hpp
///	ade
///
///	Purpose:
///	Define interfaces and building blocks for an equation graph
///

#include <memory>

#include "ade/shape.hpp"
#include "ade/coord.hpp"

#ifndef ADE_INTERFACE_HPP
#define ADE_INTERFACE_HPP

namespace ade
{

struct iLeaf;

struct iFunctor;

/// Interface to travel through graph, treating iLeaf and iFunctor differently
struct iTraveler
{
	virtual ~iTraveler (void) = default;

	/// Visit leaf node
	virtual void visit (iLeaf* leaf) = 0;

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

	/// Return the string representation of the tensor
	virtual std::string to_string (void) const = 0;
};

using TensptrT = std::shared_ptr<iTensor>;

using TensrefT = std::weak_ptr<iTensor>;

/// Smart pointer to iTensor ensuring non-null references
struct Tensorptr
{
	Tensorptr (iTensor& tens) :
		ptr_(&tens) {}

	Tensorptr (iTensor* tens) : ptr_(tens)
	{
		if (nullptr == tens)
		{
			err::fatal("cannot create nodeptr with nullptr");
		}
	}

	Tensorptr (TensptrT tens) : ptr_(tens)
	{
		if (nullptr == tens)
		{
			err::fatal("cannot create nodeptr with nullptr");
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

}

#endif // ADE_INTERFACE_HPP
