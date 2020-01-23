
#include "eteq/etens.hpp"
#include "eteq/shaper.hpp"
#include "eteq/observable.hpp"

#ifndef ETEQ_DEPEND_HPP
#define ETEQ_DEPEND_HPP

namespace eteq
{

// Proxy functor node that encapsulates operational dependency
template <typename T>
struct Depends final : public Observable
{
	static Depends<T>* get (ObsptrT dependee, const ETensorsT<T>& dependencies)
	{
		if (dependencies.empty())
		{
			teq::fatal("cannot depend on nothing");
		}

		return new Depends<T>(dependee, dependencies);
	}

	~Depends (void)
	{
		dependee_->unsubscribe(this);
	}

	/// Return deep copy of this Depends
	Depends<T>* clone (void) const
	{
		return static_cast<Depends<T>*>(clone_impl());
	}

	Depends (Depends<T>&& other) = delete;

	Depends<T>& operator = (const Depends<T>& other) = delete;

	Depends<T>& operator = (Depends<T>&& other) = delete;

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
		dependee_->update_child(arg, index);
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
		dependee_->uninitialize();
	}

private:
	Depends (ObsptrT dependee, const ETensorsT<T>& dependencies) :
		dependee_(dependee), dependencies_(dependencies)
	{
		dependee_->subscribe(this);
	}

	Depends (const Depends<T>& other) :
		dependee_(other.dependee_),
		dependencies_(other.dependencies_)
	{
		dependee_->subscribe(this);
	}

	teq::iTensor* clone_impl (void) const override
	{
		return new Depends<T>(*this);
	}

	ObsptrT dependee_;

	ETensorsT<T> dependencies_;
};

}

#endif // ETEQ_DEPEND_HPP
