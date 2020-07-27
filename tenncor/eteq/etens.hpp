//
/// etens.hpp
/// eteq
///
/// Purpose:
/// Typed tensor pointer transport object
///

#include "eteq/variable.hpp"

#ifndef ETEQ_ETENS_HPP
#define ETEQ_ETENS_HPP

namespace eteq
{

template <typename T>
struct ETensor
{
	ETensor (void) = default;

	ETensor (teq::TensptrT tens,
		eigen::CtxptrT ctx = eigen::global_context()) :
		ctx_(ctx)
	{
		if (nullptr != ctx && nullptr != tens)
		{
			registry_ = &ctx->registry_;
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
			ctx->eval_->evaluate(device, {tens});
			return (T*) tens->device().data();
		}
		return nullptr;
	}

	eigen::CtxptrT get_context (void) const
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
			ctx_ = eigen::CtxptrT(nullptr);
			registry_->erase(this);
			registry_ = nullptr;
		}
	}

	mutable eigen::CtxrefT ctx_;

	mutable eigen::TensRegistryT* registry_ = nullptr;
};

template <typename T>
struct EVariable final : public ETensor<T>
{
	EVariable (void) = default;

	EVariable (VarptrT<T> vars, eigen::CtxptrT ctx = eigen::global_context()) :
		ETensor<T>(vars, ctx) {}

	friend bool operator == (const EVariable<T>& l, const EVariable<T>& r)
	{
		return l.get() == r.get();
	}

	friend bool operator != (const EVariable<T>& l, const EVariable<T>& r)
	{
		return l.get() != r.get();
	}

	operator VarptrT<T>() const
	{
		return std::static_pointer_cast<Variable<T>>(teq::TensptrT(*this));
	}

	Variable<T>* operator-> () const
	{
		return static_cast<Variable<T>*>(this->get());
	}
};

/// Type of typed functor arguments
template <typename T>
using ETensorsT = std::vector<ETensor<T>>;

template <typename T>
using EVariablesT = std::vector<EVariable<T>>;

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
