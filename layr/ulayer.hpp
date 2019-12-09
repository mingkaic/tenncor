///
/// ulayer.hpp
/// layr
///
/// Purpose:
/// Implement generic layer that applies unary functions
/// these functions don't store any data
///

#include "eteq/generated/api.hpp"

#include "layr/layer.hpp"

#ifndef LAYR_ULAYER_HPP
#define LAYR_ULAYER_HPP

namespace layr
{

struct ULayer;

/// Function that takes corresponding unary layer and node
using UnaryF = std::function<LinkptrT(LinkptrT,LinkptrT)>;

/// ULayer parameter label
const std::string uparam_key = "uparam";

/// Builder implementation for activation layer
struct ULayerBuilder final : public iLayerBuilder
{
	ULayerBuilder (std::string act_type, std::string label) :
		utype_(act_type), label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override
	{
		if (target == uparam_key)
		{
			params_ = eteq::to_link<PybindT>(tens);
			return;
		}
		logs::warnf("attempt to create ulayer "
			"with unknown tensor `%s` of label `%s`",
			tens->to_string().c_str(), target.c_str());
	}

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override {} // activation has no sublayer

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	std::string utype_;

	LinkptrT params_ = nullptr;

	std::string label_;
};

/// Identifier for sigmoid activation layer
const std::string sigmoid_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "sigmoid",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(sigmoid_layer_key, label);
});

/// Identifier for tanh activation layer
const std::string tanh_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "tanh",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(tanh_layer_key, label);
});

/// Identifier for relu activation layer
const std::string relu_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "relu",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(relu_layer_key, label);
});

/// Identifier for softmax activation layer
const std::string softmax_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "softmax",
[](std::string labels) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(softmax_layer_key, labels);
});

/// Identifier for max pooling layer
const std::string maxpool2d_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "maxpool",
[](std::string labels) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(maxpool2d_layer_key, labels);
});

/// Identifier for mean pooling layer
const std::string meanpool2d_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "meanpool",
[](std::string labels) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(meanpool2d_layer_key, labels);
});

/// Softmax layer connection function that extracts
/// transformation parameter from layer and apply to input
LinkptrT softmax_from_param (LinkptrT input, LinkptrT params);

LinkptrT maxpool_from_param (LinkptrT input, LinkptrT params);

LinkptrT meanpool_from_param (LinkptrT input, LinkptrT params);

/// Map unary layer identifier to connection function
const std::unordered_map<std::string,UnaryF> unaries =
{
	{sigmoid_layer_key,
		[](LinkptrT input, LinkptrT params)
		{ return tenncor::sigmoid<PybindT>(input); }},
	{tanh_layer_key,
		[](LinkptrT input, LinkptrT params)
		{ return tenncor::tanh<PybindT>(input); }},
	{relu_layer_key,
		[](LinkptrT input, LinkptrT params)
		{ return tenncor::relu<PybindT>(input); }},
	{softmax_layer_key, softmax_from_param},
	{maxpool2d_layer_key, maxpool_from_param},
	{meanpool2d_layer_key, meanpool_from_param},
};

/// Layer implementation to apply activation and pooling functions
struct ULayer final : public iLayer
{
	ULayer (const std::string& ulayer_type, LinkptrT params,
		const std::string& label = "") :
		label_(label),
		utype_(ulayer_type),
		unary_(estd::must_getf(unaries, ulayer_type,
			"failed to find unary function `%s`", ulayer_type.c_str())),
		params_(params)
	{
		if (nullptr == params)
		{
			params_ = eteq::make_constant_scalar<PybindT>(0, {});
		}
		tag(params_->get_tensor(), LayerId(uparam_key));
	}

	ULayer (const ULayer& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	ULayer& operator = (const ULayer& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	ULayer (ULayer&& other) = default;

	ULayer& operator = (ULayer&& other) = default;

	/// Return deep copy of this layer with prefixed label
	ULayer* clone (std::string label_prefix = "") const
	{
		return static_cast<ULayer*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	size_t get_ninput (void) const override
	{
		return 0;
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		return 0;
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return utype_;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		return {params_->get_tensor()};
	}

	/// Implementation of iLayer
	LinkptrT connect (LinkptrT input) const override
	{
		auto output = unary_(input, params_);
		recursive_tag(output->get_tensor(), {
			input->get_tensor().get(),
		}, LayerId());
		return output;
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new ULayer(*this, label_prefix);
	}

	void copy_helper (const ULayer& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.get_label();
		utype_ = other.utype_;
		unary_ = other.unary_;
		params_ = LinkptrT(other.params_->clone());
		tag(params_->get_tensor(), LayerId(uparam_key));
	}

	std::string label_;

	std::string utype_;

	UnaryF unary_;

	LinkptrT params_;
};

/// Smart pointer of unary layer
using UnaryptrT = std::shared_ptr<ULayer>;

/// Return activation layer using sigmoid
UnaryptrT sigmoid (std::string label = "");

/// Return activation layer using tanh
UnaryptrT tanh (std::string label = "");

/// Return activation layer using relu
UnaryptrT relu (std::string label = "");

/// Return activation layer using softmax of specified dimension
UnaryptrT softmax (teq::RankT dim, std::string label = "");

/// Return pooling layer using max aggregation
UnaryptrT maxpool2d (
	std::pair<teq::DimT,teq::DimT> dims = {0, 1},
	std::string label = "");

/// Return pooling layer using mean aggregation
UnaryptrT meanpool2d (
	std::pair<teq::DimT,teq::DimT> dims = {0, 1},
	std::string label = "");

}

#endif // LAYR_ULAYER_HPP
