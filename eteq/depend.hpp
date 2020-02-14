
#include "eteq/etens.hpp"
#include "eteq/shaper.hpp"
#include "eteq/observable.hpp"

#ifndef ETEQ_DEPEND_HPP
#define ETEQ_DEPEND_HPP

namespace eteq
{

const std::string depname = "DEPEND";

// Proxy functor node that encapsulates operational dependency
struct Depends final : public Observable
{
	static Depends* get (ObsptrT dependee, const teq::TensptrsT& dependencies)
	{
		if (dependencies.empty())
		{
			teq::fatal("cannot depend on nothing");
		}

		return new Depends(dependee, dependencies);
	}

	~Depends (void)
	{
		dependee_->unsubscribe(this);
	}

	/// Return deep copy of this Depends
	Depends* clone (void) const
	{
		return static_cast<Depends*>(clone_impl());
	}

	Depends (Depends&& other) = delete;

	Depends& operator = (const Depends& other) = delete;

	Depends& operator = (Depends&& other) = delete;

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return dependee_->shape();
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return dependee_->to_string();
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return dependee_->ls_attrs();
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return dependee_->get_attr(attr_name);
	}

	/// Implementation of iAttributed
	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override
	{
		dependee_->add_attr(attr_key, std::move(attr_val));
	}

	/// Implementation of iAttributed
	void rm_attr (std::string attr_key) override
	{
		dependee_->rm_attr(attr_key);
	}

	/// Implementation of iFunctor
	teq::Opcode get_opcode (void) const override
	{
		return dependee_->get_opcode();
	}

	/// Implementation of iFunctor
	teq::TensptrsT get_children (void) const override
	{
		teq::TensptrsT out = dependee_->get_children();
		out.insert(out.end(), dependencies_.begin(), dependencies_.end());
		return out;
	}

	/// Implementation of iFunctor
	void update_child (teq::TensptrT arg, size_t index) override
	{
		size_t ndependee = dependee_->get_children().size();
		if (index < ndependee)
		{
			dependee_->update_child(arg, index);
			return;
		}
		index = ndependee - index;
		if (index >= dependencies_.size())
		{
			teq::fatalf("cannot modify dependency %d "
				"when there are only %d dependencies",
				index, dependencies_.size());
		}
		if (arg != dependencies_[index])
		{
			uninitialize();
			if (auto f = dynamic_cast<Observable*>(dependencies_[index].get()))
			{
				f->unsubscribe(this);
			}
			teq::Shape nexshape = arg->shape();
			teq::Shape curshape = dependencies_[index]->shape();
			if (false == nexshape.compatible_after(curshape, 0))
			{
				teq::fatalf("cannot update child %d to argument with "
					"incompatible shape %s (requires shape %s)",
					index, nexshape.to_string().c_str(),
					curshape.to_string().c_str());
			}
			dependencies_[index] = arg;
			if (auto f = dynamic_cast<Observable*>(arg.get()))
			{
				f->subscribe(this);
			}
		}
	}

	/// Implementation of iTensor
	teq::iDeviceRef& device (void) override
	{
		return dependee_->device();
	}

	/// Implementation of iTensor
	const teq::iDeviceRef& device (void) const override
	{
		return dependee_->device();
	}

	/// Implementation of iData
	size_t type_code (void) const override
	{
		return dependee_->type_code();
	}

	/// Implementation of iData
	std::string type_label (void) const override
	{
		return dependee_->type_label();
	}

	/// Implementation of iData
	size_t nbytes (void) const override
	{
		return dependee_->nbytes();
	}

	/// Implementation of Observable
	bool has_data (void) const override
	{
		return dependee_->has_data();
	}

	/// Implementation of Observable
	void uninitialize (void) override
	{
		if (dependee_->has_data())
		{
			dependee_->uninitialize();
			for (auto& parent : this->subs_)
			{
				parent->uninitialize();
			}
		}
	}

	ObsptrT get_deps (void) const
	{
		return dependee_;
	}

private:
	Depends (ObsptrT dependee, const teq::TensptrsT& dependencies) :
		dependee_(dependee), dependencies_(dependencies)
	{
		dependee_->subscribe(this);
	}

	Depends (const Depends& other) :
		dependee_(other.dependee_),
		dependencies_(other.dependencies_)
	{
		dependee_->subscribe(this);
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new Depends(*this);
	}

	ObsptrT dependee_;

	teq::TensptrsT dependencies_;
};

}

#endif // ETEQ_DEPEND_HPP
