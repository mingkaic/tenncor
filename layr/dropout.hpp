
#include "teq/ilayer.hpp"

#include "eteq/generated/api.hpp"

#include "layr/init.hpp"

#ifndef LAYR_DROPOUT_HPP
#define LAYR_DROPOUT_HPP

namespace layr
{

/// Mask subgraph label
const std::string dropout_mask_key = "mask";

/// Layer implementation to apply dropout masks when training
template <typename T=PybindT>
struct Dropout final : public iLayer
{
	Dropout (T prob, teq::Shape shape, const std::string& label) :
		iLayer(teq::ShapeSignature(shape)),
		label_(label)
	{
		auto p = eteq::make_constant_scalar<T>(prob, shape);
		mask_ = tenncor::rand_binom_one(p) / p;
		output_ = connect(this->input_);
	}

	Dropout (const Dropout& other, std::string label_prefix = "") :
		iLayer(other)
	{
		copy_helper(other, label_prefix);
	}

	Dropout& operator = (const Dropout& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	Dropout (Dropout&& other) = default;

	Dropout& operator = (Dropout&& other) = default;

	/// Return deep copy of this layer with prefixed label
	Dropout* clone (std::string label_prefix = "") const
	{
		return static_cast<Dropout*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	teq::ShapeSignature get_output_sign (void) const override
	{
		return output_->shape_sign();
	}

	/// Implementation of iLayer
	teq::TensptrT get_output (void) const override
	{
		return output_->get_tensor();
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return "dropout";
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_storage (void) const override
	{
		return {mask_->get_tensor()};
	}

	/// Implementation of iLayer
	teq::TensptrT connect (teq::TensptrT input) const override
	{
		return eteq::to_link<T>(input) * mask_; // todo: deactivate dropout layer when predicting
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new Dropout(*this, label_prefix);
	}

	void copy_helper (const Dropout& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		mask_ = other.mask_->clone(); // todo: recurse copy
		output_ = connect(this->input_);
	}

	std::string label_;

	eteq::LinkptrT<T> mask_;

	eteq::LinkptrT<T> output_;
};

}

#endif // LAYR_DROPOUT_HPP
