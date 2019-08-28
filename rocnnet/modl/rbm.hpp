#include "rocnnet/modl/dense.hpp"
#include "rocnnet/modl/activations.hpp"

#ifndef MODL_RBM_HPP
#define MODL_RBM_HPP

namespace modl
{

const std::string hidden_key = "hidden";

const std::string visible_key = "visible";

struct RBMBuilder final : public iLayerBuilder
{
	RBMBuilder (std::string label) : label_(label) {}

	void set_tensor (ade::TensptrT tens) override {} // rbm has no tensor

	void set_sublayer (LayerptrT layer) override
	{
		layers_.push_back(layer);
	}

	LayerptrT build (void) const override;

private:
	std::vector<LayerptrT> layers_;

	std::string label_;
};

const std::string rbm_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "rbm",
[](ade::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, rbm_layer_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<RBMBuilder>(label);
});

struct RBM final : public iLayer
{
	RBM (ade::DimT nhidden, ade::DimT nvisible,
		ActivationptrT activation,
		eqns::InitF<PybindT> weight_init,
		eqns::InitF<PybindT> bias_init,
		const std::string& label) :
		label_(label),
		hidden_(std::make_shared<Dense>(
			nhidden, nvisible, weight_init, bias_init, hidden_key)),
		activation_(activation)
	{
		auto hidden_contents = hidden_->get_contents();
		auto weight = hidden_contents[0];
		auto hbias = hidden_contents[1];
		ead::NodeptrT<PybindT> vbias = nullptr;

		if (bias_init)
		{
			vbias = bias_init(ade::Shape({nvisible}), visible_key + "_" + bias_key);
		}
		visible_ = std::make_shared<Dense>(tenncor::transpose(
			ead::NodeConverters<PybindT>::to_node(weight)), vbias, visible_key);

		auto hidden_subs = hidden_->get_contents();
		for (auto& sub : hidden_subs)
		{
			tag(sub, LayerId(hidden_->get_ltype(),
				hidden_->get_label(), 0));
		}

		auto visible_subs = visible_->get_contents();
		for (auto& sub : visible_subs)
		{
			tag(sub, LayerId(visible_->get_ltype(),
				visible_->get_label(), 1));
		}

		auto activation_subs = activation_->get_contents();
		for (auto& sub : activation_subs)
		{
			tag(sub, LayerId(activation_->get_ltype(),
				activation_->get_label(), 2));
		}
	}

	RBM (DenseptrT hidden, DenseptrT visible,
		ActivationptrT activation, std::string label) :
		label_(label),
		hidden_(hidden),
		visible_(visible),
		activation_(activation)
	{
		auto hidden_subs = hidden_->get_contents();
		for (auto& sub : hidden_subs)
		{
			tag(sub, LayerId(hidden_->get_ltype(),
				hidden_->get_label(), 0));
		}

		auto visible_subs = visible_->get_contents();
		for (auto& sub : visible_subs)
		{
			tag(sub, LayerId(visible_->get_ltype(),
				visible_->get_label(), 1));
		}

		auto activation_subs = activation_->get_contents();
		for (auto& sub : activation_subs)
		{
			tag(sub, LayerId(activation_->get_ltype(),
				activation_->get_label(), 2));
		}
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

	RBM* clone (std::string label_prefix = "") const
	{
		return static_cast<RBM*>(this->clone_impl(label_prefix));
	}

	size_t get_ninput (void) const override
	{
		return hidden_->get_ninput();
	}

	size_t get_noutput (void) const override
	{
		return hidden_->get_noutput();
	}

	std::string get_ltype (void) const override
	{
		return rbm_layer_key;
	}

	std::string get_label (void) const override
	{
		return label_;
	}

	ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> visible) const override
	{
		return activation_->connect(hidden_->connect(visible));
	}

	ade::TensT get_contents (void) const override
	{
		auto out = hidden_->get_contents();
		out.push_back(visible_->get_contents()[1]);
		return out;
	}

	ead::NodeptrT<PybindT> backward_connect (ead::NodeptrT<PybindT> hidden) const
	{
		return activation_->connect(visible_->connect(hidden));
	}

private:
	RBM* clone_impl (std::string label_prefix) const override
	{
		return new RBM(*this, label_prefix);
	}

	void copy_helper (const RBM& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		hidden_ = DenseptrT(other.hidden_->clone(label_prefix));
		visible_ = DenseptrT(other.visible_->clone(label_prefix));

		auto contents = get_contents();
		for (auto content : contents)
		{
			tag(content);
		}
	}

	std::string label_;

	DenseptrT hidden_;

	DenseptrT visible_;

	ActivationptrT activation_;
};

using RBMptrT = std::shared_ptr<RBM>;

}

#endif // MODL_RBM_HPP
