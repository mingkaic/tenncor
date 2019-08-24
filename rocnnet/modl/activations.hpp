#include "ead/generated/api.hpp"

#include "rocnnet/modl/layer.hpp"

#ifndef MODL_ACTIVATIONS_HPP
#define MODL_ACTIVATIONS_HPP

namespace modl
{

struct ActivationBuilder final : public iLayerBuilder
{
	ActivationBuilder (std::string act_type, std::string label) :
		act_type_(act_type), label_(label) {}

	void set_tensor (ade::TensptrT tens) override {}

	void set_sublayer (LayerptrT layer) override {} // activation has no sublayer

	LayerptrT build (void) const override;

private:
	std::string act_type_;

	std::string label_;
};

const std::string sigmoid_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "sigmoid",
[](ade::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, sigmoid_layer_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<ActivationBuilder>(sigmoid_layer_key, label);
});

const std::string tanh_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "tanh",
[](ade::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, tanh_layer_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<ActivationBuilder>(tanh_layer_key, label);
});

const std::unordered_map<std::string,NonLinearF> activations =
{
	{sigmoid_layer_key, tenncor::sigmoid<PybindT>},
	{tanh_layer_key, tenncor::tanh<PybindT>},
};

struct ActivationLayer final : public iLayer
{
	ActivationLayer (std::string label, std::string act_type) :
		iLayer(label), act_type_(act_type),
		activation_(estd::must_getf(activations, act_type,
			"failed to find activation `%s`", act_type.c_str())),
		placeholder_(ead::make_constant_scalar<PybindT>(0, {}))
	{
		tag(placeholder_->get_tensor());
	}

	ActivationLayer* clone (void) const
	{
		return static_cast<ActivationLayer*>(this->clone_impl());
	}

	std::string get_ltype (void) const override
	{
		return act_type_;
	}

	ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> input) const override
	{
		auto out = activation_(input);
		recursive_tag(out->get_tensor(), {
			input->get_tensor().get(),
		});
		return out;
	}

	ade::TensT get_contents (void) const override
	{
		return {placeholder_->get_tensor()};
	}

	MarsarrT get_subs (void) const override
	{
		return MarsarrT{};
	}

private:
	iMarshaler* clone_impl (void) const override
	{
		return new ActivationLayer(*this);
	}

	ead::NodeptrT<PybindT> placeholder_;

	NonLinearF activation_;

	std::string act_type_;
};

LayerptrT sigmoid (std::string label);

LayerptrT tanh (std::string label);

}

#endif // MODL_ACTIVATIONS_HPP
