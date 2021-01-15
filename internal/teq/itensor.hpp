///
/// itensor.hpp
/// teq
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#ifndef TEQ_ITENSOR_HPP
#define TEQ_ITENSOR_HPP

#include <unordered_map>
#include <unordered_set>

#include "internal/teq/shape.hpp"
#include "internal/teq/itraveler.hpp"
#include "internal/teq/once.hpp"

namespace teq
{

struct iDeviceRef
{
	virtual ~iDeviceRef (void) = default;

	/// Return pointer to internal data
	virtual void* data (void) = 0;

	/// Return const pointer to internal data
	virtual const void* data (void) const = 0;

	virtual Once<void*> odata (void) = 0;

	virtual Once<const void*> odata (void) const = 0;
};

struct iMetadata
{
	virtual ~iMetadata (void) = default;

	/// Return data type encoding
	virtual size_t type_code (void) const = 0;

	/// Return data type label (for better readability)
	virtual std::string type_label (void) const = 0;

	/// Return number of bytes for a single entry for this type
	virtual size_t type_size (void) const = 0;

	/// Return tensor version denoting its state
	virtual size_t state_version (void) const = 0;
};

/// Interface of traversible and differentiable nodes with shape information
struct iTensor : public fmts::iStringable
{
	virtual ~iTensor (void) = default;

	iTensor* clone (void) const
	{
		return this->clone_impl();
	}

	/// Obtain concrete information on either leaf or functor implementations
	virtual void accept (iTraveler& visiter) = 0;

	/// Return device reference to this tensor, device references belongs to
	/// some data-evaluation object and hold the data associated with this node
	virtual iDeviceRef& device (void) = 0;

	virtual const iDeviceRef& device (void) const = 0;

	/// Return metadata encapsulation
	virtual const iMetadata& get_meta (void) const = 0;

	/// Return the shape of the data
	virtual Shape shape (void) const = 0;

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

template <typename TS> // todo: use with concept tensptr_range
void multi_visit (iTraveler& traveler, const TS& tensors)
{
	for (auto tensor : tensors)
	{
		if (nullptr != tensor)
		{
			tensor->accept(traveler);
		}
	}
}

}

#endif // TEQ_ITENSOR_HPP
