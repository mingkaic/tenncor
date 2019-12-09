
#include "eteq/generated/api.hpp"

#include "layr/layer.hpp"

#ifndef LAYR_DROPOUT_HPP
#define LAYR_DROPOUT_HPP

namespace layr
{

/// Mask subgraph label
const std::string dropout_mask_key = "mask";

/// Builder implementation for fully connected layer
struct DropoutBuilder final : public iLayerBuilder
{
	DropoutBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override
	{
		if (target == dropout_mask_key)
		{
			mask_ = eteq::to_link<PybindT>(tens);
			return;
		}
		logs::warnf("attempt to create dropout layer "
			"with unknown tensor `%s` of label `%s`",
			tens->to_string().c_str(), target.c_str());
	}

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override {} // dense has no sublayer

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	LinkptrT mask_ = nullptr;

	std::string label_;
};

/// Identifier for fully connected layer
const std::string dropout_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "dropout",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<DropoutBuilder>(label);
});

/// Layer implementation to apply dropout masks when training
struct Dropout final : public iLayer
{
	Dropout (PybindT prob, const std::string& label) : label_(label)
	{
		auto p = eteq::make_constant_scalar<PybindT>(prob);
		mask_ = tenncor::rand_binom_one(p) / p;
		tag(mask_->get_tensor(), LayerId(dropout_mask_key));
	}

	Dropout (LinkptrT mask, const std::string& label) : label_(label), mask_(mask)
	{
		tag(mask_->get_tensor(), LayerId(dropout_mask_key));
	}

	Dropout (const Dropout& other, std::string label_prefix = "")
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
	size_t get_ninput (void) const override
	{
		return mask_->shape().at(1);
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		return mask_->shape().at(0);
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return dropout_layer_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		return {mask_->get_tensor()};
	}

	/// Implementation of iLayer
	LinkptrT connect (LinkptrT input) const override
	{
		auto output = input * mask_; // todo: deactivate dropout layer when predicting
		recursive_tag(output->get_tensor(), {
			input->get_tensor().get(),
		}, LayerId());
		return output;
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new Dropout(*this, label_prefix);
	}

	void copy_helper (const Dropout& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		mask_ = LinkptrT(other.mask_->clone()); // todo: recurse copy
		tag(mask_->get_tensor(), LayerId(dropout_weight_key));
	}

	std::string label_;

	LinkptrT mask_;
};

}

#endif // LAYR_DROPOUT_HPP
