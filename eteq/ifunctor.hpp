//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#include "teq/iopfunc.hpp"

#include "eigen/generated/opcode.hpp"

#include "eteq/link.hpp"
#include "eteq/observable.hpp"

#ifndef ETEQ_IFUNCTOR_HPP
#define ETEQ_IFUNCTOR_HPP

namespace eteq
{

/// Functor implementation of operable functor of Eigen operators
template <typename T>
struct iFunctor : public teq::iOperableFunc, public Observable<Functor<T>*>
{
	virtual ~iFunctor (void) = default;

	/// Return deep copy of this Functor
	iFunctor<T>* clone (void) const
	{
		return static_cast<iFunctor<T>*>(this->clone_impl());
	}

	/// Implementation of iTensor
	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(this);
	}

	/// Implementation of iData
	size_t type_code (void) const override
	{
		return egen::get_type<T>();
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return egen::name_type(egen::get_type<T>());
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return sizeof(T) * shape().n_elems();
	}

	/// Return true if functor has never been initialized or
	/// was uninitialized, otherwise functor can return data
	virtual bool is_uninit (void) const = 0;
};

/// Functor's node wrapper
template <typename T>
struct FuncLink final : public iLink<T>
{
	FuncLink (std::shared_ptr<iFunctor<T>> func) : func_(func)
	{
		if (func == nullptr)
		{
			logs::fatal("cannot link a null func");
		}
	}

	/// Return deep copy of this instance (with a copied functor)
	FuncLink<T>* clone (void) const
	{
		return static_cast<FuncLink<T>*>(clone_impl());
	}

	/// Implementation of iLink<T>
	T* data (void) const override
	{
		return (T*) func_->data();
	}

	/// Implementation of iLink<T>
	void update (void) override
	{
		func_->update();
	}

	/// Implementation of iLink<T>
	teq::TensptrT get_tensor (void) const override
	{
		return func_;
	}

	/// Implementation of iLink<T>
	bool has_data (void) const override
	{
		return false == func_->is_uninit();
	}

private:
	iLink<T>* clone_impl (void) const override
	{
		return new FuncLink(std::shared_ptr<iFunctor<T>>(func_->clone()));
	}

	/// Implementation of iLink<T>
	void subscribe (Functor<T>* parent) override
	{
		func_->subscribe(parent);
	}

	/// Implementation of iLink<T>
	void unsubscribe (Functor<T>* parent) override
	{
		func_->unsubscribe(parent);
	}

	std::shared_ptr<iFunctor<T>> func_;
};

template <typename T>
LinkptrT<T> func_link (teq::TensptrT tens)
{
	return std::make_shared<FuncLink<T>>(
		std::static_pointer_cast<iFunctor<T>>(tens));
}

template <typename T>
void LinkConverter<T>::visit (teq::iFunctor* leaf)
{
	builders_.emplace(leaf, func_link<T>);
}

}

#endif // ETEQ_IFUNCTOR_HPP
