#include "eteq/generated/api.hpp"

#include "layr/layer.hpp"

#ifndef LAYR_ACTIVATIONS_HPP
#define LAYR_ACTIVATIONS_HPP

namespace layr
{

struct Activation;

using NonLinearF = std::function<eteq::NodeptrT<PybindT>(
	const Activation&,eteq::NodeptrT<PybindT>)>;

struct ActivationBuilder final : public iLayerBuilder
{
	ActivationBuilder (std::string act_type, std::string label) :
		act_type_(act_type), label_(label) {}

	void set_tensor (teq::TensptrT tens, std::string target) override {}

	void set_sublayer (LayerptrT layer) override {} // activation has no sublayer

	LayerptrT build (void) const override;

private:
	std::string act_type_;

	std::string label_;
};

const std::string sigmoid_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "sigmoid",
[](teq::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, sigmoid_layer_key, "");
},
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ActivationBuilder>(sigmoid_layer_key, "");
});

const std::string tanh_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "tanh",
[](teq::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, tanh_layer_key, "");
},
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ActivationBuilder>(tanh_layer_key, "");
});

const std::string softmax_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "softmax",
[](teq::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, softmax_layer_key, label);
},
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ActivationBuilder>(softmax_layer_key, extra_info);
});

eteq::NodeptrT<PybindT> softmax_from_layer (const Activation& layer,
	eteq::NodeptrT<PybindT> input);

const std::unordered_map<std::string,NonLinearF> activations =
{
	{sigmoid_layer_key,
		[](const Activation& layer, eteq::NodeptrT<PybindT> input)
		{ return tenncor::sigmoid<PybindT>(input); }},
	{tanh_layer_key,
		[](const Activation& layer, eteq::NodeptrT<PybindT> input)
		{ return tenncor::tanh<PybindT>(input); }},
	{softmax_layer_key, softmax_from_layer},
};

struct Activation final : public iLayer
{
	Activation (const std::string& act_type, const std::string& extra_info = "") :
		label_(extra_info),
		act_type_(act_type),
		activation_(estd::must_getf(activations, act_type,
			"failed to find activation `%s`", act_type.c_str())),
		placeholder_(eteq::make_constant_scalar<PybindT>(0, {}))
	{
		tag(placeholder_->get_tensor(), LayerId());
	}

	Activation (const Activation& other,
		std::string label_prefix = "") :
		label_(label_prefix + other.get_label()),
		act_type_(other.act_type_),
		activation_(other.activation_),
		placeholder_(eteq::make_constant_scalar<PybindT>(0, {}))
	{
		tag(placeholder_->get_tensor(), LayerId());
	}

	Activation& operator = (const Activation& other) = default;

	Activation (Activation&& other) = default;

	Activation& operator = (Activation&& other) = default;


	Activation* clone (std::string label_prefix = "") const
	{
		return static_cast<Activation*>(this->clone_impl(label_prefix));
	}

	size_t get_ninput (void) const override
	{
		return 0;
	}

	size_t get_noutput (void) const override
	{
		return 0;
	}

	std::string get_ltype (void) const override
	{
		return act_type_;
	}

	std::string get_label (void) const override
	{
		return label_;
	}

	eteq::NodeptrT<PybindT> connect (eteq::NodeptrT<PybindT> input) const override
	{
		auto out = activation_(*this, input);
		recursive_tag(out->get_tensor(), {
			input->get_tensor().get(),
		}, LayerId());
		return out;
	}

	teq::TensT get_contents (void) const override
	{
		return {placeholder_->get_tensor()};
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new Activation(*this, label_prefix);
	}

	std::string label_;

	std::string act_type_;

	NonLinearF activation_;

	eteq::NodeptrT<PybindT> placeholder_;
};

using ActivationptrT = std::shared_ptr<Activation>;

ActivationptrT sigmoid (void);

ActivationptrT tanh (void);

ActivationptrT softmax (teq::RankT dim);

}

#endif // LAYR_ACTIVATIONS_HPP
