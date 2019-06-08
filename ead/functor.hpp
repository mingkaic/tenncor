#include "ade/opfunc.hpp"

#include "ead/generated/opcode.hpp"

#include "ead/funcarg.hpp"
#include "ead/constant.hpp"
#include "ead/operator.hpp"

#ifndef EAD_FUNCTOR_HPP
#define EAD_FUNCTOR_HPP

namespace ead
{

template <typename T>
struct Functor final : public ade::iOperableFunc
{
	static Functor<T>* get (ade::Opcode opcode, ArgsT<T> args);

	static Functor<T>* get (Functor<T>&& other)
	{
		return new Functor<T>(std::move(other));
	}

	Functor (const Functor<T>& other) = delete;

	Functor<T>& operator = (const Functor<T>& other) = delete;

	Functor<T>& operator = (Functor<T>&& other) = delete;

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iFunctor
	ade::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	const ade::ArgsT& get_children (void) const override
	{
		return args_;
	}

	/// Implementation of iFunctor
	void update_child (ade::FuncArg& arg, size_t index) override
	{
		ade::Shape arg_shape = arg.shape();
		if (false == arg_shape.compatible_after(shape_, 0))
		{
			logs::fatalf("cannot update child %d to argument with "
				"incompatible shape %s (requires shape %s)",
				index, arg_shape.to_string().c_str(),
				shape_.to_string().c_str());
		}
		args_[index] = arg;
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		return out_->assign();
	}

	/// Implementation of iOperableFunc
	void* raw_data (void) override
	{
		return out_->get_ptr();
	}

	/// Implementation of iOperableFunc
	size_t type_code (void) const override
	{
		return age::get_type<T>();
	}

private:
	Functor (EigenptrT<T> out,
		ade::Opcode opcode, ade::Shape shape, ade::ArgsT args) :
		out_(out), opcode_(opcode), shape_(shape), args_(args) {}

	Functor (Functor<T>&& other) = default;

	EigenptrT<T> out_;

	/// Operation encoding
	ade::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	ade::Shape shape_;

	/// Tensor arguments (and children)
	ade::ArgsT args_;
};

template <typename T>
struct FunctorNode final : public iNode<T>
{
	FunctorNode (std::shared_ptr<Functor<T>> f) : func_(f) {}

	T* data (void) override
	{
		return (T*) func_->raw_data();
	}

	void update (void) override
	{
		func_->update();
	}

	ade::TensptrT get_tensor (void) override
	{
		return func_;
	}

private:
	std::shared_ptr<Functor<T>> func_;
};

template <typename T>
Functor<T>* Functor<T>::get (ade::Opcode opcode, ArgsT<T> args)
{
	static bool registered = register_builder<Functor<T>,T>(
		[](ade::TensptrT tens)
		{
			return std::make_shared<FunctorNode<T>>(
				std::static_pointer_cast<Functor<T>>(tens));
		});
	assert(registered);

	if (0 == args.size())
	{
		logs::fatalf("cannot perform %s with no arguments",
			opcode.name_.c_str());
	}

	ade::Shape shape = args[0].shape();
	for (size_t i = 1, n = args.size(); i < n; ++i)
	{
		ade::Shape ishape = args[i].shape();
		if (false == ishape.compatible_after(shape, 0))
		{
			logs::fatalf("cannot perform %s with incompatible shapes %s "
				"and %s", opcode.name_.c_str(), shape.to_string().c_str(),
				ishape.to_string().c_str());
		}
	}

	ade::ArgsT input_args;
	std::vector<OpArg<T>> tmaps;
	for (FuncArg<T>& arg : args)
	{
		NodeptrT<T> node = arg.get_node();
		ade::TensptrT tensor = node->get_tensor();
		CoordptrT coorder = arg.get_coorder();
		input_args.push_back(ade::FuncArg(
			tensor,
			arg.get_shaper(),
			arg.map_io(),
			coorder));
		tmaps.push_back(OpArg<T>{
			node->data(),
			tensor->shape(),
			coorder.get()
		});
	}
	EigenptrT<T> eigen;
	age::typed_exec<T>((age::_GENERATED_OPCODE) opcode.code_,
		shape, eigen, tmaps);
	return new Functor<T>(eigen,
		opcode, shape, input_args);
}

template <typename T>
NodeptrT<T> make_functor (ade::Opcode opcode, ArgsT<T> args)
{
	return std::make_shared<FunctorNode<T>>(
		std::shared_ptr<Functor<T>>(Functor<T>::get(opcode, args))
	);
}

}

#endif // EAD_FUNCTOR_HPP
