///
/// ileaf.hpp
/// eteq
///
/// Purpose:
/// Define interfaces and building blocks for an equation graph
///

#include "teq/ileaf.hpp"

#include "eigen/generated/dtype.hpp"
#include "eigen/eigen.hpp"

#include "eteq/link.hpp"

#ifndef ETEQ_ILEAF_HPP
#define ETEQ_ILEAF_HPP

namespace eteq
{

/// iLeaf extension of TEQ iLeaf containing Eigen data objects
template <typename T>
struct iLeaf : public teq::iLeaf
{
	virtual ~iLeaf (void) = default;

	iLeaf<T>* clone (void) const
	{
		return static_cast<iLeaf<T>*>(this->clone_impl());
	}

	/// Implementation of iTensor
	void accept (teq::iTraveler& visiter) override
	{
		visiter.visit(*this);
	}

	/// Implementation of iTensor
	teq::Shape shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iData
	void* data (void) override
	{
		return data_.data();
	}

	/// Implementation of iData
	const void* data (void) const override
	{
		return data_.data();
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

protected:
	iLeaf (T* data, teq::Shape shape) :
		data_(eigen::make_tensmap(data, shape)),
		shape_(shape) {}

	/// Data Source
	eigen::TensorT<T> data_;

	/// Shape utility to avoid excessive conversion between data_.dimensions()
	teq::Shape shape_;
};

/// Leaf tensor wrapper
template <typename T>
struct LeafLink final : public iLink<T>
{
	LeafLink (std::shared_ptr<iLeaf<T>> leaf) : leaf_(leaf)
	{
		if (leaf == nullptr)
		{
			logs::fatal("cannot link a null leaf");
		}
	}

	/// Return deep copy of this instance (with a copied constant)
	LeafLink<T>* clone (void) const
	{
		return static_cast<LeafLink<T>*>(clone_impl());
	}

	/// Implementation of iAttributed
	std::vector<std::string> ls_attrs (void) const override
	{
		return {};
	}

	/// Implementation of iAttributed
	const marsh::iObject* get_attr (std::string attr_name) const override
	{
		return nullptr;
	}

	/// Implementation of iAttributed
	void add_attr (std::string attr_key, marsh::ObjptrT&& attr_val) override {}

	/// Implementation of iAttributed
	void rm_attr (std::string attr_key) override {}

	/// Implementation of iLink<T>
	teq::TensptrT get_tensor (void) const override
	{
		return leaf_;
	}

	/// Implementation of iSignature
	bool can_build (void) const override
	{
		return true;
	}

	/// Implementation of iSignature
	teq::DataptrT build_data (void) const override
	{
		return leaf_;
	}

	/// Implementation of iSignature
	teq::ShapeSignature shape_sign (void) const override
	{
		teq::Shape shape = leaf_->shape();
		return teq::ShapeSignature(
			std::vector<teq::DimT>(shape.begin(), shape.end()));
	}

private:
	LeafLink (const LeafLink<T>& other) = default;

	iLink<T>* clone_impl (void) const override
	{
		return new LeafLink(std::shared_ptr<iLeaf<T>>(leaf_->clone()));
	}

	/// Implementation of iLink<T>
	void subscribe (Functor<T>* parent) override {}

	/// Implementation of iLink<T>
	void unsubscribe (Functor<T>* parent) override {}

	std::shared_ptr<iLeaf<T>> leaf_;
};

}

#endif // ETEQ_ILEAF_HPP
