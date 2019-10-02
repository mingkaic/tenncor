#include "layr/dense.hpp"
#include "layr/activations.hpp"

#ifndef LAYR_RBM_HPP
#define LAYR_RBM_HPP

namespace layr
{

const std::string hidden_key = "hidden";

const std::string visible_key = "visible";

struct RBMBuilder final : public iLayerBuilder
{
	RBMBuilder (std::string label) : label_(label) {}

	void set_tensor (teq::TensptrT tens, std::string target) override {} // rbm has no tensor

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
[](teq::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, rbm_layer_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<RBMBuilder>(label);
});

struct RBM final : public iLayer
{
	RBM (teq::DimT nhidden, teq::DimT nvisible,
		ActivationptrT activation,
		layr::InitF<PybindT> weight_init,
		layr::InitF<PybindT> bias_init,
		const std::string& label) :
		label_(label),
		hidden_(std::make_shared<Dense>(
			nhidden, nvisible, weight_init, bias_init, hidden_key)),
		activation_(activation)
	{
		auto hidden_contents = hidden_->get_contents();
		auto weight = hidden_contents[0];
		auto hbias = hidden_contents[1];
		NodeptrT vbias = nullptr;

		if (bias_init)
		{
			vbias = bias_init(teq::Shape({nvisible}), bias_key);
		}
		visible_ = std::make_shared<Dense>(tenncor::transpose(
			eteq::NodeConverters<PybindT>::to_node(weight)), vbias, visible_key);
		tag_sublayers();
	}

	RBM (DenseptrT hidden, DenseptrT visible,
		ActivationptrT activation, std::string label) :
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

	NodeptrT connect (NodeptrT visible) const override
	{
		return activation_->connect(hidden_->connect(visible));
	}

	teq::TensT get_contents (void) const override
	{
		auto out = hidden_->get_contents();
		auto vis_contents = visible_->get_contents();
		auto act_contents = activation_->get_contents();
		out.insert(out.end(), vis_contents.begin(), vis_contents.end());
		out.insert(out.end(), act_contents.begin(), act_contents.end());
		return out;
	}

	NodeptrT backward_connect (NodeptrT hidden) const
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
		NodeptrT vbias_node = nullptr;
		if (auto vbias = other.visible_->get_contents()[1])
		{
			vbias_node = NodeptrT(eteq::NodeConverters<PybindT>::to_node(
				vbias)->clone());
		}
		visible_ = std::make_shared<Dense>(tenncor::transpose(
			eteq::NodeConverters<PybindT>::to_node(hidden_contents[0])),
			vbias_node, label_prefix + visible_key);

		activation_ = ActivationptrT(other.activation_->clone(label_prefix));
		tag_sublayers();
	}

	std::string label_;

	DenseptrT hidden_;

	DenseptrT visible_;

	ActivationptrT activation_;
};

using RBMptrT = std::shared_ptr<RBM>;

}

#endif // LAYR_RBM_HPP
