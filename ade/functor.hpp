///
/// functor.hpp
/// ade
///
/// Purpose:
/// Define functor nodes of an equation graph
///

#include "ade/ifunctor.hpp"

#ifndef ADE_FUNCTOR_HPP
#define ADE_FUNCTOR_HPP

namespace ade
{

/// Functor of the graph mapping to operators specified by opcode argument
struct Functor final : public iFunctor
{
	/// Return a Functor with with input tensor and meta arguments
	static Functor* get (Opcode opcode, ArgsT args)
	{
		if (0 == args.size())
		{
			logs::fatalf("cannot perform %s with no arguments",
				opcode.name_.c_str());
		}

		Shape shape = args[0].shape();
		for (size_t i = 1, n = args.size(); i < n; ++i)
		{
			Shape ishape = args[i].shape();
			if (false == ishape.compatible_after(shape, 0))
			{
				logs::fatalf("cannot perform %s with incompatible shapes %s "
					"and %s", opcode.name_.c_str(), shape.to_string().c_str(),
					ishape.to_string().c_str());
			}
		}
		return new Functor(opcode, shape, args);
	}

	static Functor* get (const Functor& other)
	{
		return new Functor(other);
	}

	static Functor* get (Functor&& other)
	{
		return new Functor(std::move(other));
	}

	Functor& operator = (const Functor& other) = delete;

	Functor& operator = (Functor&& other) = delete;

	/// Implementation of iTensor
	const Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return opcode_.name_;
	}

	/// Implementation of iFunctor
	Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	const ArgsT& get_children (void) const override
	{
		return args_;
	}

	/// Implementation of iFunctor
	void update_child (FuncArg arg, size_t index) override
	{
		logs::debug("ade::Functor does not allow editing of children");
	}

private:
	Functor (Opcode opcode, Shape shape, ArgsT args) :
		opcode_(opcode), shape_(shape), args_(args) {}

	Functor (const Functor& other) = default;

	Functor (Functor&& other) = default;

	/// Operation encoding
	Opcode opcode_;

	/// Shape info built at construction time according to arguments
	Shape shape_;

	/// Tensor arguments (and children)
	ArgsT args_;
};

}

#endif // ADE_FUNCTOR_HPP
