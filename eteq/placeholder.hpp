#include "teq/placeholder.hpp"

#include "eteq/functor.hpp"

#ifndef ETEQ_PLACEHOLDER_HPP
#define ETEQ_PLACEHOLDER_HPP

namespace eteq
{

/// A "dynamic" function that takes nodes OR
/// shaped array/eigen tensors as a single input
/// If its input are shaped array/eigen tensors,
/// then wrap it in variable and assign as child
/// Otherwise take node as its child
template <typename T>
struct PlaceLink final : public iLink<T>
{
	PlaceLink (teq::ShapeSignature shape, std::string label = "") :
		place_(std::make_shared<teq::Placeholder>(shape, label)) {}

	/// Return deep copy of this Functor
	PlaceLink<T>* clone (void) const
	{
		return static_cast<PlaceLink<T>*>(clone_impl());
	}

	void assign (const teq::LeafptrT& inleaf)
	{
		place_->assign(inleaf);
	}

	void assign (const OpFuncptrT<T>& infunc)
	{
		place_->assign(infunc);
	}

	void assign (const LinkptrT<T>& input)
	{
		place_->assign(input->build_data());
	}

	void assign (eigen::TensMapT<T>& input)
	{
		teq::Shape shape = eigen::get_shape(input);
		if (auto var = place_->can_build() ?
			dynamic_cast<Variable<T>*>(place_->build_data().get()) : nullptr)
		{
			if (false == shape.compatible_after(shape, 0))
			{
				logs::fatalf("assigning data shaped %s to tensor %s",
					shape.to_string().c_str(), shape.to_string().c_str());
			}
			var->assign(input);
		}
		else
		{
			assign(make_variable<T>(input.data(), shape, place_->to_string()));
		}
	}

	void assign (eigen::TensorT<T>& input)
	{
		auto tensmap = eigen::tens_to_tensmap(input);
		assign(tensmap);
	}

	void assign (teq::ShapedArr<T>& sarr)
	{
		if (auto var = place_->can_build() ?
			dynamic_cast<Variable<T>*>(place_->build_data().get()) : nullptr)
		{
			auto shape = var->shape();
			if (false == sarr.shape_.compatible_after(shape, 0))
			{
				logs::fatalf("assigning data shaped %s to tensor %s",
					sarr.shape_.to_string().c_str(), shape.to_string().c_str());
			}
			var->assign(sarr);
		}
		else
		{
			assign(make_variable<T>(
				sarr.data_.data(), sarr.shape_, place_->to_string()));
		}
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
		return place_;
	}

	/// Implementation of iSignature
	bool can_build (void) const override
	{
		return place_->can_build();
	}

	/// Implementation of iSignature
	teq::DataptrT build_data (void) const override
	{
		return place_->build_data();
	}

	/// Implementation of iSignature
	teq::ShapeSignature shape_sign (void) const override
	{
		return place_->shape_sign();
	}

private:
	PlaceLink (const PlaceLink<T>& other) = default;

	iLink<T>* clone_impl (void) const override
	{
		return new PlaceLink<T>(*this);
	}

	void subscribe (Functor<T>* parent) override {}

	void unsubscribe (Functor<T>* parent) override {}

	teq::PlaceptrT place_;
};

/// Smart pointer of placeholder nodes to preserve assign functions
template <typename T>
using PlaceLinkptrT = std::shared_ptr<PlaceLink<T>>;

}

#endif // ETEQ_PLACEHOLDER_HPP
