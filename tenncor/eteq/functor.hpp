//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#ifndef ETEQ_FUNCTOR_HPP
#define ETEQ_FUNCTOR_HPP

#include "tenncor/eteq/etens.hpp"

namespace eteq
{

#define _CHOOSE_PARSER(OPCODE)\
outshape = egen::ShapeParser<OPCODE>()(attrs, shapes);

/// Functor implementation of operable functor of Eigen operators
template <typename T>
struct Functor final : public eigen::Observable
{
	/// Return Functor given opcodes mapped to Eigen operators in operator.hpp
	/// Return nullptr if functor is redundant
	static Functor<T>* get (egen::_GENERATED_OPCODE opcode,
		teq::TensptrsT children, marsh::Maps&& attrs)
	{
		if (children.empty())
		{
			global::fatalf("cannot perform `%s` without arguments",
				egen::name_op(opcode).c_str());
		}

		teq::ShapesT shapes;
		shapes.reserve(children.size());
		std::transform(children.begin(), children.end(),
			std::back_inserter(shapes),
			[](teq::TensptrT tens) { return tens->shape(); });

		auto ctype = children.front()->get_meta().type_code();
		for (auto it = children.begin(), et = children.end();
			it != et; ++it)
		{
			auto child = *it;
			if (ctype != child->get_meta().type_code())
			{
				global::fatal("children types are not all the same");
			}
		}

		teq::Shape outshape;
		OPCODE_LOOKUP(_CHOOSE_PARSER, opcode)
		return new Functor<T>(
			opcode, outshape, children, std::move(attrs));
	}

	~Functor (void)
	{
		for (teq::TensptrT child : args_)
		{
			if (auto f = dynamic_cast<eigen::Observable*>(child.get()))
			{
				f->unsubscribe(this);
			}
		}
	}

	/// Return deep copy of this Functor
	Functor<T>* clone (void) const
	{
		return static_cast<Functor<T>*>(clone_impl());
	}

	Functor (Functor<T>&& other) = delete;

	Functor<T>& operator = (const Functor<T>& other) = delete;

	Functor<T>& operator = (Functor<T>&& other) = delete;

	/// Implementation of iTensor
	teq::Shape shape (void) const override
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
	teq::TensptrsT get_args (void) const override
	{
		return args_;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		if (index >= args_.size())
		{
			global::throw_errf("cannot replace argument %d when only "
				"there are only %d available", index, args_.size());
		}
		uninitialize();
		if (auto f = dynamic_cast<eigen::Observable*>(args_[index].get()))
		{
			f->unsubscribe(this);
		}
		teq::Shape nexshape = arg->shape();
		teq::Shape curshape = args_[index]->shape();
		if (false == nexshape.compatible_after(curshape, 0))
		{
			global::fatalf("cannot update child %d to argument with "
				"incompatible shape %s (requires shape %s)",
				index, nexshape.to_string().c_str(),
				curshape.to_string().c_str());
		}
		auto nextype = arg->get_meta().type_label();
		auto curtype = args_[index]->get_meta().type_label();
		if (curtype != nextype)
		{
			global::fatalf("cannot update child %d to argument with "
				"different type %s (requires type %s)",
				index, nextype.c_str(), curtype.c_str());
		}
		args_[index] = arg;
		if (auto f = dynamic_cast<eigen::Observable*>(arg.get()))
		{
			f->subscribe(this);
		}
	}

	/// Implementation of iTensor
	teq::iDeviceRef& device (void) override
	{
		if (false == has_data())
		{
			must_initialize();
		}
		return *ref_;
	}

	/// Implementation of iTensor
	const teq::iDeviceRef& device (void) const override
	{
		if (false == has_data())
		{
			global::fatal("cannot get device of uninitialized functor");
		}
		return *ref_;
	}

	/// Implementation of iTensor
	const teq::iMetadata& get_meta (void) const override
	{
		return meta_;
	}

	/// Implementation of Observable
	bool has_data (void) const override
	{
		return nullptr != ref_;
	}

	/// Implementation of Observable
	void uninitialize (void) override
	{
		if (has_data())
		{
			ref_ = nullptr;
			meta_.version_ = 0;
			for (auto& parent : this->subs_)
			{
				parent->uninitialize();
			}
		}
	}

	/// Implementation of Observable
	bool initialize (void) override
	{
		if (std::all_of(args_.begin(), args_.end(),
			[](teq::TensptrT child)
			{
				if (auto f = dynamic_cast<eigen::Observable*>(child.get()))
				{
					return f->has_data();
				}
				return true;
			}))
		{
			egen::typed_exec<T>((egen::_GENERATED_OPCODE) opcode_.code_,
				ref_, shape_, args_, *this);
		}
		return has_data();
	}

	/// Implementation of Observable
	void must_initialize (void) override
	{
		for (auto child : args_)
		{
			auto f = dynamic_cast<eigen::Observable*>(child.get());
			if (nullptr != f && false == f->has_data())
			{
				f->must_initialize();
			}
		}
		if (false == initialize())
		{
			global::fatal("failed to initialize");
		}
	}

	/// Implementation of Observable
	bool prop_version (size_t max_version) override
	{
		size_t des_version = 0;
		for (auto& child : args_)
		{
			des_version = std::max(des_version, child->get_meta().state_version());
		}
		// non-idempotent will want to execute regardless of version
		// and op should execute if desired version > current version, so ...
		size_t cur_version = meta_.version_;
		if (des_version <= cur_version &&
			false == egen::is_idempotent(
			(egen::_GENERATED_OPCODE) opcode_.code_))
		{
			des_version = cur_version + 1;
		}
		des_version = std::min(des_version, max_version);
		bool propped = meta_.version_ < des_version;
		if (propped)
		{
			meta_.version_ = des_version;
		}
		return propped;
	}

private:
	Functor (egen::_GENERATED_OPCODE opcode, teq::Shape shape,
		teq::TensptrsT args, marsh::Maps&& attrs) :
		eigen::Observable(args, std::move(attrs)),
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		shape_(shape), args_(args)
	{
		common_init();
	}

	Functor (const Functor<T>& other) :
		eigen::Observable(other),
		opcode_(other.opcode_),
		shape_(other.shape_),
		args_(other.args_)
	{
		common_init();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new Functor<T>(*this);
	}

	void common_init (void)
	{
#ifndef SKIP_INIT
		initialize();
#endif // SKIP_INIT
	}

	eigen::EigenptrT ref_ = nullptr;

	/// Operation encoding
	teq::Opcode opcode_;

	/// Shape info built at construction time according to arguments
	teq::Shape shape_;

	/// Tensor arguments (and children)
	teq::TensptrsT args_;

	eigen::EMetadata<T> meta_;
};

#undef _CHOOSE_PARSER

template <typename T>
using FuncptrT = std::shared_ptr<Functor<T>>;

}

#endif // ETEQ_FUNCTOR_HPP
