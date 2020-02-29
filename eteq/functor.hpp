//
/// functor.hpp
/// eteq
///
/// Purpose:
/// Eigen functor implementation of operable func
///

#include "eteq/etens.hpp"
#include "eteq/shaper.hpp"
#include "eteq/observable.hpp"

#ifndef ETEQ_FUNCTOR_HPP
#define ETEQ_FUNCTOR_HPP

namespace eteq
{

#define CHOOSE_PARSER(OPCODE)\
outshape = ShapeParser<OPCODE>().shape(attrs, shapes);

/// Functor implementation of operable functor of Eigen operators
template <typename T>
struct Functor final : public Observable
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
			if (tcode != child->type_code())
			{
				teq::fatalf("incompatible tensor types %s and %s: "
					"cross-type functors not supported yet",
					egen::name_type(tcode).c_str(),
					child->type_label().c_str());
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
			if (auto f = dynamic_cast<Observable*>(child.get()))
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

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return attrs_.ls_attrs();
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (const std::string& attr_name) const override
	{
		return attrs_.get_attr(attr_name);
	}

	/// Implementation of iAttributed
	void add_attr (const std::string& attr_key, marsh::ObjptrT&& attr_val) override
	{
		attrs_.add_attr(attr_key, std::move(attr_val));
	}

	/// Implementation of iAttributed
	void rm_attr (const std::string& attr_key) override
	{
		attrs_.rm_attr(attr_key);
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return opcode_;
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_children (void) const override
	{
		return children_;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		if (index >= children_.size())
		{
			teq::fatalf("cannot modify argument %d "
				"when there are only %d arguments",
				index, children_.size());
		}
		if (arg != children_[index])
		{
			uninitialize();
			if (auto f = dynamic_cast<Observable*>(children_[index].get()))
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
			if (auto f = dynamic_cast<Observable*>(arg.get()))
			{
				f->subscribe(this);
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
				if (auto f = dynamic_cast<Observable*>(child.get()))
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
			auto f = dynamic_cast<Observable*>(child.get());
			if (nullptr != f && false == f->has_data())
			{
				f->must_initialize();
			}
		}
		assert(initialize());
	}

private:
	Functor (egen::_GENERATED_OPCODE opcode, teq::Shape shape,
		teq::TensptrsT children, marsh::Maps&& attrs) :
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		shape_(shape), children_(children), attrs_(std::move(attrs))
	{
		common_init();
	}

	Functor (const Functor<T>& other) :
		opcode_(other.opcode_),
		shape_(other.shape_),
		children_(other.children_)
	{
		std::unique_ptr<marsh::Maps> mattr(other.attrs_.clone());
		attrs_ = std::move(*mattr);
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
			if (auto f = dynamic_cast<Observable*>(child.get()))
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

	marsh::Maps attrs_;
};

#undef CHOOSE_PARSER

template <typename T>
using FuncptrT = std::shared_ptr<Functor<T>>;

/// Return functor node given opcode and node arguments
template <typename T, typename ...ARGS>
ETensor<T> make_functor (egen::_GENERATED_OPCODE opcode,
	const teq::TensptrsT& children, ARGS... vargs);

}

#endif // ETEQ_FUNCTOR_HPP
