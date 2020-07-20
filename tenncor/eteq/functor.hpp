//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#include "eteq/etens.hpp"
#include "eteq/shaper.hpp"

#ifndef ETEQ_FUNCTOR_HPP
#define ETEQ_FUNCTOR_HPP

namespace eteq
{

#define CHOOSE_PARSER(OPCODE)\
outshape = ShapeParser<OPCODE>().shape(attrs, shapes);

const std::string dependency_key = "dependencies";

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
			teq::fatalf("cannot perform `%s` without arguments",
				egen::name_op(opcode).c_str());
		}

		teq::ShapesT shapes;
		shapes.reserve(children.size());
		egen::_GENERATED_DTYPE tcode = egen::get_type<T>();
		for (teq::TensptrT child : children)
		{
			if (tcode != child->get_meta().type_code())
			{
				teq::fatalf("incompatible tensor types %s and %s: "
					"cross-type functors not supported yet",
					egen::name_type(tcode).c_str(),
					child->get_meta().type_label().c_str());
			}
			shapes.push_back(child->shape());
		}

		teq::Shape outshape;
		OPCODE_LOOKUP(CHOOSE_PARSER, opcode)
		return new Functor<T>(
			opcode, outshape, children, std::move(attrs));
	}

	~Functor (void)
	{
		for (teq::TensptrT child : children_)
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
		return children_;
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_dependencies (void) const override
	{
		auto deps = children_;
		if (auto deps_attr = dynamic_cast<const marsh::iArray*>(
			this->get_attr(dependency_key)))
		{
			deps_attr->foreach(
				[&](size_t, const marsh::iObject* obj)
				{
					if (auto dep = dynamic_cast<const teq::TensorObj*>(obj))
					{
						deps.push_back(dep->get_tensor());
					}
				});
		}
		return deps;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		auto deps_attr = dynamic_cast<teq::TensArrayT*>(
			this->get_attr(dependency_key));
		size_t ndeps = children_.size();
		if (nullptr != deps_attr)
		{
			ndeps += deps_attr->contents_.size();
		}
		if (index >= ndeps)
		{
			teq::fatalf("cannot modify dependency %d "
				"when there are only %d dependencies",
				index, ndeps);
		}
		if (index < children_.size())
		{
			if (arg != children_[index])
			{
				uninitialize();
				if (auto f = dynamic_cast<eigen::Observable*>(children_[index].get()))
				{
					f->unsubscribe(this);
				}
				teq::Shape nexshape = arg->shape();
				teq::Shape curshape = children_[index]->shape();
				if (false == nexshape.compatible_after(curshape, 0))
				{
					teq::fatalf("cannot update child %d to argument with "
						"incompatible shape %s (requires shape %s)",
						index, nexshape.to_string().c_str(),
						curshape.to_string().c_str());
				}
				children_[index] = arg;
				if (auto f = dynamic_cast<eigen::Observable*>(arg.get()))
				{
					f->subscribe(this);
				}
			}
		}
		else
		{
			size_t idep = index - children_.size();
			if (arg != deps_attr->contents_[idep]->get_tensor())
			{
				deps_attr->contents_[idep] =
					std::make_unique<teq::TensorObj>(arg);
			}
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
			teq::fatal("cannot get device of uninitialized functor");
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
		if (std::all_of(children_.begin(), children_.end(),
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
				ref_, shape_, children_, *this);
		}
		return has_data();
	}

	/// Implementation of Observable
	void must_initialize (void) override
	{
		for (auto child : children_)
		{
			auto f = dynamic_cast<eigen::Observable*>(child.get());
			if (nullptr != f && false == f->has_data())
			{
				f->must_initialize();
			}
		}
		assert(initialize());
	}

	/// Implementation of Observable
	bool prop_version (size_t max_version) override
	{
		size_t des_version = 0;
		for (auto& child : children_)
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
		teq::TensptrsT children, marsh::Maps&& attrs) :
		eigen::Observable(std::move(attrs)),
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		shape_(shape), children_(children)
	{
		common_init();
	}

	Functor (const Functor<T>& other) :
		eigen::Observable(other),
		opcode_(other.opcode_),
		shape_(other.shape_),
		children_(other.children_)
	{
		common_init();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new Functor<T>(*this);
	}

	void common_init (void)
	{
		for (teq::TensptrT child : children_)
		{
			if (auto f = dynamic_cast<eigen::Observable*>(child.get()))
			{
				f->subscribe(this);
			}
		}
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
	teq::TensptrsT children_;

	eigen::EMetadata<T> meta_;
};

#undef CHOOSE_PARSER

template <typename T>
using FuncptrT = std::shared_ptr<Functor<T>>;

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
teq::TensptrT make_functor (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children,  ARGS... vargs);

template <typename T, typename ...ARGS>
ETensor<T> make_functor (eteq::ETensRegistryT& registry,
	egen::_GENERATED_OPCODE opcode, const teq::TensptrsT& children,
	ARGS... vargs);

}

#endif // ETEQ_FUNCTOR_HPP
