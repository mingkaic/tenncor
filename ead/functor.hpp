#include "ade/ifunctor.hpp"

#include "ead/generated/opmap.hpp"

#include "ead/funcarg.hpp"
#include "ead/constant.hpp"
#include "ead/operator.hpp"

#ifndef EAD_FUNCTOR_HPP
#define EAD_FUNCTOR_HPP

namespace ead
{

template <typename T>
struct Functor final : public ade::iFunctor
{
	static Functor<T>* get (ade::Opcode opcode, ArgsT<T> args)
	{
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
		return new Functor<T>(age::typed_exec<T>(
			(age::_GENERATED_OPCODE) opcode.code_, shape, tmaps),
			opcode, shape, input_args);	
	}

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

	void update (void)
	{
		return out_->assign();
	}

	T* data (void)
	{
		return out_->get_ptr();
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
struct FuncNode final : public iNode<T>
{
	FuncNode (std::shared_ptr<Functor<T>> f) : func_(f) {}

	T* data (void) override
	{
		return func_->data();
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
NodeptrT<T> make_functor (ade::Opcode opcode, ArgsT<T> args)
{
	return std::make_shared<FuncNode<T>>(
		std::shared_ptr<Functor<T>>(Functor<T>::get(opcode, args))
	);
}

}

#endif // EAD_FUNCTOR_HPP
