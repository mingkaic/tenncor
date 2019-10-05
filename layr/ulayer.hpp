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
using UnaryF = std::function<NodeptrT(
	const ULayer&,NodeptrT)>;

/// Builder implementation for activation layer
struct ULayerBuilder final : public iLayerBuilder
{
	ULayerBuilder (std::string act_type, std::string label) :
		utype_(act_type), label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override {}

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override {} // activation has no sublayer

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	std::string utype_;

	std::string label_;
};

/// Identifier for sigmoid activation layer
const std::string sigmoid_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "sigmoid",
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(sigmoid_layer_key, "");
});

/// Identifier for tanh activation layer
const std::string tanh_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "tanh",
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(tanh_layer_key, "");
});

/// Identifier for softmax activation layer
const std::string softmax_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "softmax",
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(softmax_layer_key, extra_info);
});

/// Identifier for max pooling layer
const std::string maxpool2d_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "maxpool",
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(maxpool2d_layer_key, extra_info);
});

/// Identifier for mean pooling layer
const std::string meanpool2d_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "meanpool",
[](std::string extra_info) -> LBuilderptrT
{
	return std::make_shared<ULayerBuilder>(meanpool2d_layer_key, extra_info);
});

/// Softmax layer connection function that extracts
/// transformation parameter from layer and apply to input
NodeptrT softmax_from_layer (const ULayer& layer, NodeptrT input);

NodeptrT maxpool_from_layer (const ULayer& layer, NodeptrT input);

NodeptrT meanpool_from_layer (const ULayer& layer, NodeptrT input);

/// Map unary layer identifier to connection function
const std::unordered_map<std::string,UnaryF> unaries =
{
	{sigmoid_layer_key,
		[](const ULayer& layer, NodeptrT input)
		{ return tenncor::sigmoid<PybindT>(input); }},
	{tanh_layer_key,
		[](const ULayer& layer, NodeptrT input)
		{ return tenncor::tanh<PybindT>(input); }},
	{softmax_layer_key, softmax_from_layer},
	{maxpool2d_layer_key, maxpool_from_layer},
	{meanpool2d_layer_key, meanpool_from_layer},
};

/// Layer implementation to apply activation and pooling functions
struct ULayer final : public iLayer
{
	ULayer (const std::string& ulayer_type, const std::string& extra_info = "") :
		label_(extra_info),
		utype_(ulayer_type),
		unary_(estd::must_getf(unaries, ulayer_type,
			"failed to find unary function `%s`", ulayer_type.c_str())),
		placeholder_(eteq::make_constant_scalar<PybindT>(0, {}))
	{
		tag(placeholder_->get_tensor(), LayerId());
	}

	ULayer (const ULayer& other,
		std::string label_prefix = "") :
		label_(label_prefix + other.get_label()),
		utype_(other.utype_),
		unary_(other.unary_),
		placeholder_(eteq::make_constant_scalar<PybindT>(0, {}))
	{
		tag(placeholder_->get_tensor(), LayerId());
	}

	ULayer& operator = (const ULayer& other) = default;

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
		return {placeholder_->get_tensor()};
	}

	/// Implementation of iLayer
	NodeptrT connect (NodeptrT input) const override
	{
		auto out = unary_(*this, input);
		recursive_tag(out->get_tensor(), {
			input->get_tensor().get(),
		}, LayerId());
		return out;
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new ULayer(*this, label_prefix);
	}

	std::string label_;

	std::string utype_;

	UnaryF unary_;

	NodeptrT placeholder_;
};

/// Smart pointer of unary layer
using UnaryptrT = std::shared_ptr<ULayer>;

/// Return activation layer using sigmoid
UnaryptrT sigmoid (void);

/// Return activation layer using tanh
UnaryptrT tanh (void);

/// Return activation layer using softmax of specified dimension
UnaryptrT softmax (teq::RankT dim);

/// Return pooling layer using max aggregation
UnaryptrT maxpool2d (std::pair<teq::DimT,teq::DimT> dims = {0, 1});

/// Return pooling layer using mean aggregation
UnaryptrT meanpool2d (std::pair<teq::DimT,teq::DimT> dims = {0, 1});

}

#endif // LAYR_ULAYER_HPP
