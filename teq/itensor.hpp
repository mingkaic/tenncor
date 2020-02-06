///
/// itensor.hpp
/// teq
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include <unordered_map>
#include <unordered_set>

#include "teq/shape.hpp"

#ifndef TEQ_ITENSOR_HPP
#define TEQ_ITENSOR_HPP

namespace teq
{

struct iLeaf;

struct iFunctor;

/// Interface to travel through graph, treating iLeaf and iFunctor differently
struct iTraveler
{
	virtual ~iTraveler (void) = default;

	/// Visit leaf node
	virtual void visit (iLeaf& leaf) = 0;

	/// Visit functor node
	virtual void visit (iFunctor& func) = 0;
};

struct iDeviceRef
{
	virtual ~iDeviceRef (void) = default;

	/// Return pointer to internal data
	virtual void* data (void) = 0;

	/// Return const pointer to internal data
	virtual const void* data (void) const = 0;
};

/// Interface of traversible and differentiable nodes with shape information
struct iTensor
{
	virtual ~iTensor (void) = default;

	iTensor* clone (void) const
	{
		return this->clone_impl();
	}

	// // Keep around for debugging purposes
	// void* data (void)
	// {
	// 	return device().data();
	// }

	// // Keep around for debugging purposes
	// const void* data (void) const
	// {
	// 	return device().data();
	// }

	/// Obtain concrete information on either leaf or functor implementations
	virtual void accept (iTraveler& visiter) = 0;

	/// Return device reference to this tensor, device references
	/// belongs to session and hold the data associated with this node
	virtual iDeviceRef& device (void) = 0;

	virtual const iDeviceRef& device (void) const = 0;

	/// Return the shape of the data
	virtual Shape shape (void) const = 0;

	/// Return data type encoding
	virtual size_t type_code (void) const = 0;

	/// Return data type label (for better readability)
	virtual std::string type_label (void) const = 0;

	/// Return number of bytes in the data
	virtual size_t nbytes (void) const = 0;

	/// Return the string representation of the tensor
	virtual std::string to_string (void) const = 0;

protected:
	virtual iTensor* clone_impl (void) const = 0;
};

/// Tensor smart pointer
using TensptrT = std::shared_ptr<iTensor>;

/// Tensor weak pointers
using TensrefT = std::weak_ptr<iTensor>;

/// Vector of raw tensor pointers
using TensT = std::vector<iTensor*>;

using CTensT = std::vector<const iTensor*>;

/// Vector of tensor smart pointers
using TensptrsT = std::vector<TensptrT>;

/// Hash set of raw tensor pointers
using TensSetT = std::unordered_set<iTensor*>;

/// Hash set of tensor smart pointers
using TensptrSetT = std::unordered_set<TensptrT>;

template <typename V>
using TensMapT = std::unordered_map<iTensor*,V>;

template <typename V>
using CTensMapT = std::unordered_map<const iTensor*,V>;

}

#endif // TEQ_ITENSOR_HPP
