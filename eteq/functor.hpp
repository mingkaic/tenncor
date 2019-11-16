//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#include "teq/iopfunc.hpp"

#include "tag/locator.hpp"

#include "eigen/generated/opcode.hpp"
#include "eigen/operator.hpp"

#include "eteq/edge.hpp"

#ifndef ETEQ_FUNCTOR_HPP
#define ETEQ_FUNCTOR_HPP

namespace eteq
{

/// Functor implementation of operable functor of Eigen operators
template <typename T>
struct Functor final : public teq::iOperableFunc
{
	/// Return Functor given opcodes mapped to Eigen operators in operator.hpp
	static Functor<T>* get (teq::Opcode opcode, ArgsT<T> args);

	/// Return Functor move of other
	static Functor<T>* get (Functor<T>&& other)
	{
		return new Functor<T>(std::move(other));
	}

	Functor (const Functor<T>& other) = delete;

	Functor<T>& operator = (const Functor<T>& other) = delete;

	Functor<T>& operator = (Functor<T>&& other) = delete;

	/// Implementation of iTensor
	const teq::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	teq::CEdgesT get_children (void) const override
	{
		return teq::CEdgesT(args_.begin(), args_.end());
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		teq::Shape nexshape = arg->shape();
		teq::Shape curshape = args_[index].argshape();
		if (false == nexshape.compatible_after(curshape, 0))
		{
			logs::fatalf("cannot update child %d to argument with "
				"incompatible shape %s (requires shape %s)",
				index, nexshape.to_string().c_str(),
				curshape.to_string().c_str());
		}
		static_cast<Edge<T>*>(&args_[index])->set_tensor(arg);
		// todo: warn of data destruction
		uninitialize();
	}

	/// Implementation of iOperableFunc
	void update (void) override
	{
		if (is_uninit())
		{
			initialize();
		}
		out_->assign();
	}

	/// Implementation of iData
	void* data (void) override
	{
		if (is_uninit())
		{
			initialize();
		}
		return out_->get_ptr();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		if (is_uninit())
		{
			logs::fatal("cannot get data of uninitialized functor");
		}
		return out_->get_ptr();
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

	/// Return true if functor has never been initialized or
	/// was uninitialized, otherwise functor can return data
	bool is_uninit (void) const
	{
		return nullptr == out_;
	}

	/// Removes internal Eigen data object
	void uninitialize (void)
	{
		out_ = nullptr;
	}

	/// Populate internal Eigen data object
	void initialize (void)
	{
		egen::typed_exec<T>((egen::_GENERATED_OPCODE) opcode_.code_,
			shape_, out_, eigen::EigenEdgesT<T>(args_.begin(), args_.end()));
	}

private:
	Functor (teq::Opcode opcode, teq::Shape shape, ArgsT<T> args) :
		opcode_(opcode), shape_(shape), args_(args)
	{
#ifdef FINIT_ON_BUILD
		initialize();
#endif // FINIT_ON_BUILD
	}

	Functor (Functor<T>&& other) = default;

	eigen::EigenptrT<T> out_ = nullptr;

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::Shape shape_;

	/// Tensor arguments (and children)
	ArgsT<T> args_;
};

/// Functor's node wrapper
template <typename T>
struct FunctorNode final : public iNode<T>
{
	FunctorNode (std::shared_ptr<Functor<T>> f) : func_(f) {}

	/// Return deep copy of this instance (with a copied functor)
	FunctorNode<T>* clone (void) const
	{
		return static_cast<FunctorNode<T>*>(clone_impl());
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
	iNode<T>* clone_impl (void) const override
	{
		auto args = func_->get_children();
		ArgsT<T> input_args;
		input_args.reserve(args.size());
		std::transform(args.begin(), args.end(),
			std::back_inserter(input_args),
			[](const teq::iEdge& arg) -> Edge<T>
			{
				return *static_cast<const Edge<T>*>(&arg);
			});
		return new FunctorNode(std::shared_ptr<Functor<T>>(
			Functor<T>::get(func_->get_opcode(), input_args)));
	}

private:
	std::shared_ptr<Functor<T>> func_;
};

template <typename T>
Functor<T>* Functor<T>::get (teq::Opcode opcode, ArgsT<T> args)
{
	static bool registered = register_builder<Functor<T>,T>(
		[](teq::TensptrT tens)
		{
			return std::make_shared<FunctorNode<T>>(
				std::static_pointer_cast<Functor<T>>(tens));
		});
	assert(registered);

	size_t nargs = args.size();
	if (0 == nargs)
	{
		logs::fatalf("cannot perform `%s` with no arguments",
			opcode.name_.c_str());
	}

	teq::Shape shape = args[0].shape();
	for (size_t i = 1, n = nargs; i < n; ++i)
	{
		teq::Shape ishape = args[i].shape();
		if (false == ishape.compatible_after(shape, 0))
		{
			std::stringstream location;
			for (size_t i = 0; i < nargs; ++i)
			{
				location << "[" << i << "]\n"
					<< tag::display_location(args[i].get_tensor()) << "\n";
			}
			logs::fatalf("cannot perform `%s` with incompatible shapes %s "
				"and %s: childrens\n%s", opcode.name_.c_str(),
				shape.to_string().c_str(), ishape.to_string().c_str(),
				location.str().c_str());
		}
	}

	return new Functor<T>(opcode, shape, args);
}

/// Return functor node given opcode and node arguments
template <typename T>
NodeptrT<T> make_functor (teq::Opcode opcode, ArgsT<T> args)
{
	return std::make_shared<FunctorNode<T>>(
		std::shared_ptr<Functor<T>>(Functor<T>::get(opcode, args))
	);
}

}

#endif // ETEQ_FUNCTOR_HPP
