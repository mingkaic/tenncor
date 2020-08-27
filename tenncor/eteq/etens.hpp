//
/// etens.hpp
/// eteq
///
/// Purpose:
/// Typed tensor pointer transport object
///

#ifndef ETEQ_ETENS_HPP
#define ETEQ_ETENS_HPP

#include "internal/eigen/eigen.hpp"

namespace eteq
{

using TensRegistryT = std::unordered_map<void*,teq::TensptrT>;

void set_reg (TensRegistryT* reg, global::CfgMapptrT ctx = global::context());

TensRegistryT& get_reg (const global::CfgMapptrT& ctx = global::context());

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

	ETensor (const ETensor& other)
	{
		copy(other);
	}

	ETensor (ETensor&& other)
	{
		copy(other);
		other.cleanup();
	}

	ETensor& operator = (const ETensor& other)
	{
		if (this != &other)
		{
			cleanup();
			copy(other);
		}
		return *this;
	}

	ETensor& operator = (ETensor&& other)
	{
		if (this != &other)
		{
			cleanup();
			copy(other);
			other.cleanup();
		}
		return *this;
	}

	friend bool operator == (const ETensor& l, const std::nullptr_t&)
	{
		return l.get() == nullptr;
	}

	friend bool operator == (const std::nullptr_t&, const ETensor& r)
	{
		return nullptr == r.get();
	}

	friend bool operator != (const ETensor& l, const std::nullptr_t&)
	{
		return l.get() != nullptr;
	}

	friend bool operator != (const std::nullptr_t&, const ETensor& r)
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

	global::CfgMapptrT get_context (void) const
	{
		if (ctx_.expired())
		{
			return nullptr;
		}
		return ctx_.lock();
	}

	template <typename T>
	T* data (void)
	{
		auto tens = get();
		assert(tens->get_meta().type_code() == egen::get_type<T>());
		return (T*) tens->device().data();
	}

	template <typename T>
	T* calc (teq::TensSetT ignored = {},
		size_t max_version = std::numeric_limits<size_t>::max())
	{
		if (auto ctx = get_context())
		{
			auto tens = get();
			eigen::Device device(max_version);
			teq::get_eval(ctx).evaluate(device, {tens});
			return data<T>();
		}
		return nullptr;
	}

private:
	void copy (const ETensor& other)
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

using ETensorsT = std::vector<ETensor>;

teq::TensptrsT to_tensors (const ETensorsT& etensors);

}

#endif // ETEQ_ETENS_HPP
