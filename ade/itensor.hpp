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

/// Coordinate mapper and tensor pair
struct MappedTensor final
{
	MappedTensor (CoordPtrT mapper, TensptrT tensor) :
		mapper_(mapper), tensor_(tensor)
	{
		if (tensor_ == nullptr)
		{
			err::fatal("cannot map a null tensor");
		}
	}

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
	TensptrT tensor_;
};

}

#endif // ADE_INTERFACE_HPP
