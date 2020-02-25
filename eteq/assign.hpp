
#include "eteq/etens.hpp"
#include "eteq/shaper.hpp"
#include "eteq/observable.hpp"

#ifndef ETEQ_ASSIGN_HPP
#define ETEQ_ASSIGN_HPP

namespace eteq
{

template <typename T>
struct Assign final : public Observable
{
	static Assign<T>* get (egen::_GENERATED_OPCODE opcode,
		VarptrT<T> target, teq::TensptrsT children)
	{
		if (children.empty())
		{
			teq::fatalf("cannot perform `%s` without arguments",
				egen::name_op(opcode).c_str());
		}

		return new Assign<T>(opcode, target, children);
	}

	~Assign (void)
	{
		for (teq::TensptrT child : children_)
		{
			if (auto f = dynamic_cast<Observable*>(child.get()))
			{
				f->unsubscribe(this);
			}
		}
	}

	/// Return deep copy of this Assign
	Assign<T>* clone (void) const
	{
		return static_cast<Assign<T>*>(clone_impl());
	}

	Assign (Assign<T>&& other) = delete;

	Assign<T>& operator = (const Assign<T>& other) = delete;

	Assign<T>& operator = (Assign<T>&& other) = delete;

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return target_->shape();
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
	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return attrs_.get_attr(attr_name);
	}

	/// Implementation of iAttributed
	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		attrs_.add_attr(attr_key, std::move(attr_val));
	}

	/// Implementation of iAttributed
	void rm_attr (std::string attr_key) override
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
		teq::TensptrsT out = {target_};
		out.insert(out.end(), children_.begin(), children_.end());
		return out;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		if (0 == index)
		{
			teq::fatal("cannot reassign target of assignment (index 0)");
		}
		--index;
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
			teq::fatal("cannot get device of uninitialized Assign");
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
		return sizeof(T) * shape().n_elems();
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
				ref_, shape(), get_children(), *this);
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
		initialize();
	}

private:
	Assign (egen::_GENERATED_OPCODE opcode, VarptrT<T> target,
		teq::TensptrsT children) :
		opcode_(teq::Opcode{egen::name_op(opcode), opcode}),
		target_(target), children_(children)
	{
		common_init();
	}

	Assign (const Assign<T>& other) :
		opcode_(other.opcode_),
		target_(other.target_),
		children_(other.children_)
	{
		std::unique_ptr<marsh::Maps> mattr(other.attrs_.clone());
		attrs_ = std::move(*mattr);
		common_init();
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new Assign<T>(*this);
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
	VarptrT<T> target_;

	/// Tensor arguments (and children)
	teq::TensptrsT children_;

	marsh::Maps attrs_;
};

}

#endif // ETEQ_ASSIGN_HPP
