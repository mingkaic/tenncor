
#ifndef EIGEN_OBSERVABLE_HPP
#define EIGEN_OBSERVABLE_HPP

#include "internal/eigen/emeta.hpp"

namespace eigen
{

struct iMutableLeaf : public teq::iLeaf
{
	virtual ~iMutableLeaf (void) = default;

	/// Increment metadata version
	virtual void upversion (size_t version) = 0;
};

struct Observable : public teq::iFunctor
{
	Observable (void) = default;

	Observable (marsh::Maps&& attrs) : attrs_(std::move(attrs)) {}

	Observable (const Observable& other)
	{
		std::unique_ptr<marsh::Maps> mattr(other.attrs_.clone());
		attrs_ = std::move(*mattr);
	}

	Observable (Observable&& other) = default;

	Observable& operator = (const Observable& other)
	{
		if (this != &other)
		{
			std::unique_ptr<marsh::Maps> mattr(other.attrs_.clone());
			attrs_ = std::move(*mattr);
		}
		return *this;
	}

	Observable& operator = (Observable&& other) = default;

	virtual ~Observable (void) = default;

	void subscribe (Observable* sub)
	{
		subs_.emplace(sub);
	}

	void unsubscribe (Observable* sub)
	{
		subs_.erase(sub);
	}

	virtual bool has_data (void) const = 0;

	/// Removes internal data object
	virtual void uninitialize (void) = 0;

	/// Best effort internal data object initialization
	/// Return true if initialized, otherwise false
	/// Will not recursively initialize children
	virtual bool initialize (void) = 0;

	/// Do or die populate internal data object, will recurse
	virtual void must_initialize (void) = 0;

	/// Update metadata version to a desired version
	/// Desired version is the maximum version of all functor children
	/// and can be no higher than max_version
	/// Return true if metadata version is updated to the desired version
	/// Metadata version is only updated to the desired version
	/// if the desired version > current version
	/// Returning true also denotes that the functor should update data
	virtual bool prop_version (size_t max_version) = 0;

	/// Implementation of iAttributed
	types::StringsT ls_attrs (void) const override
	{
		return attrs_.ls_attrs();
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (const std::string& attr_key) const override
	{
		return attrs_.get_attr(attr_key);
	}

	/// Implementation of iAttributed
	marsh::iObject* get_attr (const std::string& attr_key) override
	{
		return attrs_.get_attr(attr_key);
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

	/// Implementation of iAttributed
	size_t size (void) const override
	{
		return attrs_.size();
	}

protected:
	std::unordered_set<Observable*> subs_;

private:
	marsh::Maps attrs_;
};

using ObsptrT = std::shared_ptr<Observable>;

}

#endif // EIGEN_OBSERVABLE_HPP
