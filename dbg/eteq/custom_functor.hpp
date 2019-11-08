///
/// custom_functor.hpp
/// dbg
///
/// Purpose:
/// Define custom functor version of eteq functor
///

#include "teq/iopfunc.hpp"

#include "eigen/generated/opcode.hpp"
#include "eigen/operator.hpp"

#include "eteq/funcarg.hpp"

#ifndef DBG_CUSTOM_FUNCTOR_HPP
#define DBG_CUSTOM_FUNCTOR_HPP

namespace dbg
{

/// Custom functor to assign DataMap to Eigen tensor output
template <typename T>
using CustomOpF = std::function<void(eigen::TensorT<T>&,const DataMapT<T>&)>;

/// Functor that runs a custom functor instead of Eigen operators
template <typename T>
struct CustomFunctor final : public teq::iOperableFunc
{
	/// Return a CustomFunctor with input function and meta arguments
	static CustomFunctor<T>* get (CustomOpF<T> op, eteq::ArgsT<T> args);

	CustomFunctor (const CustomFunctor<T>& other) = default;

	CustomFunctor (CustomFunctor<T>&& other) = default;

	CustomFunctor<T>& operator = (const CustomFunctor<T>& other) = delete;

	CustomFunctor<T>& operator = (CustomFunctor<T>&& other) = delete;

	/// Implementation of iTensor
	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return "DBG_CUSTOM_FUNCTOR";
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return teq::Opcode{this->to_string(), 0};
	}

	/// Implementation of iFunctor
	teq::CEdgesT get_children (void) const override
	{
		return teq::CEdgesT(args_.begin(), args_.end());
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		logs::fatal("cannot modify custom functor");
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		op_(out_, args_);
	}

	/// Implementation of iData
	void* data (void) override
	{
		return out_.data();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		return out_.data();
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
		return sizeof(T) * shape_.n_elems();
	}

private:
	CustomFunctor (CustomOpF<T> op, teq::Shape shape, ArgsT<T> args) :
		out_(eigen::shape_convert(shape)),
		op_(op), shape_(shape), args_(args) {}

	eigen::TensorT<T> out_;

	CustomOpF<T> op_;

	/// Shape info built at construction time according to arguments
	teq::Shape shape_;

	/// Tensor arguments (and children)
	ArgsT<T> args_;
};

/// CustomFunctor's node wrapper
template <typename T>
struct CustomFunctorNode final : public eteq::iNode<T>
{
	CustomFunctorNode (std::shared_ptr<CustomFunctor<T>> f) : func_(f) {}

	/// Return deep copy of this instance (with a copied functor)
	CustomFunctorNode<T>* clone (void) const
	{
		return static_cast<CustomFunctorNode<T>*>(clone_impl());
	}

	/// Implementation of iNode<T>
	T* data (void) override
	{
		return (T*) func_->data();
	}

	/// Implementation of iNode<T>
	void update (void) override
	{
		func_->update();
	}

	/// Implementation of iNode<T>
	teq::TensptrT get_tensor (void) const override
	{
		return func_;
	}

protected:
	eteq::iNode<T>* clone_impl (void) const override
	{
		return new CustomFunctorNode(
			std::make_shared<CustomFunctor<T>>(*func_));
	}

private:
	std::shared_ptr<CustomFunctor<T>> func_;
};

template <typename T>
CustomFunctor<T>* CustomFunctor<T>::get (CustomOpF<T> op, eteq::ArgsT<T> args)
{
	static bool registered = eteq::register_builder<CustomFunctor<T>,T>(
		[](teq::TensptrT tens)
		{
			return std::make_shared<CustomFunctorNode<T>>(
				std::static_pointer_cast<CustomFunctor<T>>(tens));
		});
	assert(registered);

	size_t nargs = args.size();
	if (0 == nargs)
	{
		logs::fatal("cannot create custom functor without args");
	}

	teq::Shape shape = args[0].shape();
	for (size_t i = 1, n = nargs; i < n; ++i)
	{
		teq::Shape ishape = args[i].shape();
		if (false == ishape.compatible_after(shape, 0))
		{
			logs::fatalf("cannot create custom functor with "
				"incompatible shapes %s and %s",
				shape.to_string().c_str(),
				ishape.to_string().c_str());
		}
	}
	return new CustomFunctor<T>(op, shape, args);
}

/// Return custom functor node given custom function and arguments
template <typename T>
eteq::NodeptrT<T> make_functor (CustomOpF<T> op, eteq::ArgsT<T> args)
{
	return std::make_shared<CustomFunctorNode<T>>(
		std::shared_ptr<CustomFunctor<T>>(CustomFunctor<T>::get(op, args))
	);
}

}

#endif // DBG_CUSTOM_FUNCTOR_HPP
