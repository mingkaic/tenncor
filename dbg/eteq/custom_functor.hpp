#include "teq/iopfunc.hpp"

#include "eteq/generated/opcode.hpp"

#include "eteq/funcarg.hpp"
#include "eteq/constant.hpp"
#include "eteq/operator.hpp"

#ifndef DBG_CUSTOM_FUNCTOR_HPP
#define DBG_CUSTOM_FUNCTOR_HPP

namespace dbg
{

template <typename T>
using DataMapT = std::vector<eteq::OpArg<T>>;

template <typename T>
using CustomOpF = std::function<void(eteq::TensorT<T>&,const DataMapT<T>&)>;

template <typename T>
struct CustomFunctor final : public teq::iOperableFunc
{
	static CustomFunctor<T>* get (CustomOpF<T> op, eteq::ArgsT<T> args);

	static CustomFunctor<T>* get (CustomFunctor<T>&& other)
	{
		return new CustomFunctor<T>(std::move(other));
	}

	CustomFunctor (const CustomFunctor<T>& other) = delete;

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
	const teq::ArgsT& get_children (void) const override
	{
		return args_;
	}

	/// Implementation of iFunctor
	void update_child (teq::FuncArg arg, size_t index) override
	{
		logs::fatal("cannot modify custom functor");
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		DataMapT<T> datamaps;
		for (const teq::FuncArg& arg : args_)
		{
			auto tens = arg.get_tensor();
			auto coorder = static_cast<eteq::CoordMap*>(arg.get_coorder().get());
			datamaps.push_back(eteq::OpArg<T>{
				eteq::TO_NODE(tens)->data(),
				tens->shape(),
				coorder
			});
		}
		op_(out_, datamaps);
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
	CustomFunctor (CustomOpF<T> op, teq::Shape shape, teq::ArgsT args) :
		out_(eteq::shape_convert(shape)),
		op_(op), shape_(shape), args_(args) {}

	CustomFunctor (CustomFunctor<T>&& other) = default;

	eteq::TensorT<T> out_;

	CustomOpF<T> op_;

	/// Shape info built at construction time according to arguments
	teq::Shape shape_;

	/// Tensor arguments (and children)
	teq::ArgsT args_;
};

template <typename T>
struct CustomFunctorNode final : public eteq::iNode<T>
{
	CustomFunctorNode (std::shared_ptr<CustomFunctor<T>> f) : func_(f) {}

	T* data (void) override
	{
		return (T*) func_->data();
	}

	void update (void) override
	{
		func_->update();
	}

	teq::TensptrT get_tensor (void) const override
	{
		return func_;
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

	teq::ArgsT input_args;
	input_args.reserve(nargs);
	std::transform(args.begin(), args.end(),
		std::back_inserter(input_args),
		[](eteq::FuncArg<T>& arg)
		{
			return teq::FuncArg(
				arg.get_tensor(),
				arg.get_shaper(),
				arg.map_io(),
				arg.get_coorder());
		});
	return new CustomFunctor<T>(op, shape, input_args);
}

template <typename T>
eteq::NodeptrT<T> make_functor (CustomOpF<T> op, eteq::ArgsT<T> args)
{
	return std::make_shared<CustomFunctorNode<T>>(
		std::shared_ptr<CustomFunctor<T>>(CustomFunctor<T>::get(op, args))
	);
}

}

#endif // DBG_CUSTOM_FUNCTOR_HPP
