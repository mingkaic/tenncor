///
/// rbm.hpp
/// layr
///
/// Purpose:
/// Implement restricted boltzmann layer
///

#include "layr/dense.hpp"
#include "layr/ulayer.hpp"

#ifndef LAYR_RBM_HPP
#define LAYR_RBM_HPP

namespace layr
{

/// Hidden fully connected layer label
const std::string hidden_key = "hidden";

/// Visible fully connected layer label
const std::string visible_key = "visible";

/// Builder implementation for restricted boltzmann layer
struct RBMBuilder final : public iLayerBuilder
{
	RBMBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override {} // rbm has no tensor

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override
	{
		layers_.push_back(layer);
	}

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	std::vector<LayerptrT> layers_;

	std::string label_;
};

/// Identifier for restricted boltzmann machine
const std::string rbm_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "rbm",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<RBMBuilder>(label);
});

/// Layer implemnetation that connects forward and backward
/// through 2 Dense layers sharing a weight
struct RBM final : public iLayer
{
	RBM (teq::DimT nhidden, teq::DimT nvisible,
		UnaryptrT activation,
		layr::InitF<PybindT> weight_init,
		layr::InitF<PybindT> bias_init,
		const std::string& label) :
		label_(label),
		hidden_(std::make_shared<Dense>(nhidden, teq::Shape({nvisible}),
			weight_init, bias_init, nullptr, hidden_key)),
		activation_(activation)
	{
		auto hidden_contents = hidden_->get_contents();
		auto weight = hidden_contents[0];
		auto hbias = hidden_contents[1];
		teq::TensptrT vbias = nullptr;

		if (bias_init)
		{
			vbias = bias_init(teq::Shape({nvisible}), dense_bias_key);
		}
		visible_ = std::make_shared<Dense>(tenncor::transpose(
			eteq::to_link<PybindT>(weight))->get_tensor(),
			vbias, nullptr, visible_key);
		tag_sublayers();

		placeholder_connect();
	}

	RBM (DenseptrT hidden, DenseptrT visible,
		UnaryptrT activation, const std::string& label) :
		label_(label),
		hidden_(hidden),
		visible_(visible),
		activation_(activation)
	{
		tag_sublayers();
	}

	RBM (const RBM& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	RBM& operator = (const RBM& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	RBM (RBM&& other) = default;

	RBM& operator = (RBM&& other) = default;

	/// Return deep copy of this layer with prefixed label
	RBM* clone (std::string label_prefix = "") const
	{
		return static_cast<RBM*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	size_t get_ninput (void) const override
	{
		return hidden_->get_ninput();
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		return hidden_->get_noutput();
	}

	/// Implementation of iLayer
	teq::ShapeSignature get_input_sign (void) const override
	{
		return hidden_->get_input_sign();
	}

	/// Implementation of iLayer
	teq::ShapeSignature get_output_sign (void) const override
	{
		return hidden_->get_output_sign();
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return rbm_layer_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		auto out = hidden_->get_contents();
		auto vis_contents = visible_->get_contents();
		auto act_contents = activation_->get_contents();
		out.insert(out.end(), vis_contents.begin(), vis_contents.end());
		out.insert(out.end(), act_contents.begin(), act_contents.end());
		return out;
	}

	/// Implementation of iLayer
	LinkptrT connect (LinkptrT visible) const override
	{
		return activation_->connect(hidden_->connect(visible));
	}

	/// Return visible reconstruction from hidden
	LinkptrT backward_connect (LinkptrT hidden) const
	{
		return activation_->connect(visible_->connect(hidden));
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new RBM(*this, label_prefix);
	}

	void tag_sublayers (void)
	{
		auto hidden_subs = hidden_->get_contents();
		for (auto& sub : hidden_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(hidden_->get_ltype(),
					hidden_->get_label(), 0));
			}
		}

		auto visible_subs = visible_->get_contents();
		for (auto& sub : visible_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(visible_->get_ltype(),
					visible_->get_label(), 1));
			}
		}

		auto activation_subs = activation_->get_contents();
		for (auto& sub : activation_subs)
		{
			tag(sub, LayerId(activation_->get_ltype(),
				activation_->get_label(), 2));
		}
	}

	void copy_helper (const RBM& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		hidden_ = DenseptrT(other.hidden_->clone(label_prefix));
		auto hidden_contents = hidden_->get_contents();
		teq::TensptrT vbias_cpy = nullptr;
		auto other_vis_contents = other.visible_->get_contents();
		if (auto vbias = other_vis_contents[1])
		{
			vbias_cpy = teq::TensptrT(vbias->clone());
		}
		visible_ = std::make_shared<Dense>(tenncor::transpose(
			eteq::to_link<PybindT>(hidden_contents[0]))->get_tensor(),
			vbias_cpy, eteq::to_link<PybindT>(
				other_vis_contents[2] == nullptr ?
				nullptr : teq::TensptrT(other_vis_contents[2]->clone())),
			label_prefix + visible_key);

		activation_ = UnaryptrT(other.activation_->clone(label_prefix));
		tag_sublayers();

		this->input_ = nullptr;
		this->placeholder_connect();
	}

	std::string label_;

	DenseptrT hidden_;

	DenseptrT visible_;

	UnaryptrT activation_;
};

/// Smart pointer of RBM layer
using RBMptrT = std::shared_ptr<RBM>;

}

#endif // LAYR_RBM_HPP
