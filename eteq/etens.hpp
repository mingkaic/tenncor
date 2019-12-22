//
/// etens.hpp
/// eteq
///
/// Purpose:
/// Typed Tensor pointer transport object
///

#include "estd/estd.hpp"

#include "teq/itensor.hpp"

#include "eigen/generated/dtype.hpp"
#include "eigen/eigen.hpp"

#ifndef ETEQ_ETENS_HPP
#define ETEQ_ETENS_HPP

namespace eteq
{

template <typename T>
struct ETensor
{
	ETensor (void) : tens_(nullptr) {}

	ETensor (teq::TensptrT tens) : tens_(tens) {}

	virtual ~ETensor (void) = default;

	operator teq::TensptrT() const
	{
		return tens_;
	}

	teq::iTensor& operator* () const
	{
		return *tens_;
	}

	teq::iTensor* operator-> () const
	{
		return tens_.get();
	}

	teq::iTensor* get (void) const
	{
		return tens_.get();
	}

private:
	teq::TensptrT tens_;
};

/// Type of typed functor arguments
template <typename T>
using ETensorsT = std::vector<ETensor<T>>;

template <typename T>
teq::TensptrsT to_tensors (const ETensorsT<T>& etensors)
{
	teq::TensptrsT tensors;
	tensors.reserve(etensors.size());
	std::transform(etensors.begin(), etensors.end(), std::back_inserter(tensors),
		[](ETensor<T> etens)
		{
			return (teq::TensptrT) etens;
		});
	return tensors;
}

}

template <typename T>
bool operator == (const eteq::ETensor<T>& l, const std::nullptr_t&)
{
	return l.get() == nullptr;
}

template <typename T>
bool operator == (const std::nullptr_t&, const eteq::ETensor<T>& r)
{
	return nullptr == r.get();
}

template <typename T>
bool operator != (const eteq::ETensor<T>& l, const std::nullptr_t&)
{
	return l.get() != nullptr;
}

template <typename T>
bool operator != (const std::nullptr_t&, const eteq::ETensor<T>& r)
{
	return nullptr != r.get();
}

#endif // ETEQ_ETENS_HPP
