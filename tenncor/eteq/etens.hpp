//
/// etens.hpp
/// eteq
///
/// Purpose:
/// Typed tensor pointer transport object
///

#include "eigen/eigen.hpp"

#ifndef ETEQ_ETENS_HPP
#define ETEQ_ETENS_HPP

namespace eteq
{

using TensRegistryT = std::unordered_map<void*,teq::TensptrT>;

void set_reg (TensRegistryT* reg, global::CfgMapptrT ctx = global::context());

TensRegistryT& get_reg (const global::CfgMapptrT& ctx = global::context());

template <typename T>
struct ETensor
{
	ETensor (void) = default;

	ETensor (teq::TensptrT tens,
		const global::CfgMapptrT& ctx = global::context()) :
		ctx_(ctx)
	{
		if (nullptr != ctx && nullptr != tens)
		{
			registry_ = &get_reg(ctx);
			registry_->emplace(this, tens);
		}
	}

	virtual ~ETensor (void)
	{
		cleanup();
	}

	ETensor (const ETensor<T>& other)
	{
		copy(other);
	}

	ETensor (ETensor<T>&& other)
	{
		copy(other);
		other.cleanup();
	}

	ETensor& operator = (const ETensor<T>& other)
	{
		if (this != &other)
		{
			cleanup();
			copy(other);
		}
		return *this;
	}

	ETensor& operator = (ETensor<T>&& other)
	{
		if (this != &other)
		{
			cleanup();
			copy(other);
			other.cleanup();
		}
		return *this;
	}

	friend bool operator == (const ETensor<T>& l, const std::nullptr_t&)
	{
		return l.get() == nullptr;
	}

	friend bool operator == (const std::nullptr_t&, const ETensor<T>& r)
	{
		return nullptr == r.get();
	}

	friend bool operator != (const ETensor<T>& l, const std::nullptr_t&)
	{
		return l.get() != nullptr;
	}

	friend bool operator != (const std::nullptr_t&, const ETensor<T>& r)
	{
		return nullptr != r.get();
	}

	operator teq::TensptrT() const
	{
		return nullptr == registry_ ? nullptr :
			estd::try_get(*registry_, (void*) this, nullptr);
	}

	teq::iTensor& operator* () const
	{
		return *get();
	}

	teq::iTensor* operator-> () const
	{
		return get();
	}

	teq::iTensor* get (void) const
	{
		return teq::TensptrT(*this).get();
	}

	T* data (void)
	{
		return (T*) get()->device().data();
	}

	T* calc (teq::TensSetT ignored = {},
		size_t max_version = std::numeric_limits<size_t>::max())
	{
		if (auto ctx = get_context())
		{
			auto tens = get();
			eigen::Device device(max_version);
			teq::get_eval(ctx).evaluate(device, {tens});
			return (T*) tens->device().data();
		}
		return nullptr;
	}

	global::CfgMapptrT get_context (void) const
	{
		if (ctx_.expired())
		{
			return nullptr;
		}
		return ctx_.lock();
	}

private:
	void copy (const ETensor<T>& other)
	{
		if ((registry_ = other.registry_))
		{
			ctx_ = other.ctx_;
			registry_->emplace(this, teq::TensptrT(other));
		}
	}

	void cleanup (void)
	{
		if (false == ctx_.expired() && nullptr != registry_)
		{
			ctx_ = global::CfgMapptrT(nullptr);
			registry_->erase(this);
			registry_ = nullptr;
		}
	}

	mutable global::CfgMaprefT ctx_;

	mutable TensRegistryT* registry_ = nullptr;
};

template <typename T>
using ETensorsT = std::vector<ETensor<T>>;

template <typename T>
teq::TensptrsT to_tensors (const ETensorsT<T>& etensors)
{
	teq::TensptrsT tensors;
	tensors.reserve(etensors.size());
	std::transform(etensors.begin(), etensors.end(),
		std::back_inserter(tensors),
		[](ETensor<T> etens)
		{
			return (teq::TensptrT) etens;
		});
	return tensors;
}

}

#endif // ETEQ_ETENS_HPP
