#include "ead/generated/api.hpp"

#include "rocnnet/eqns/init.hpp"

#include "rocnnet/modl/layer.hpp"

#ifndef MODL_DENSE_HPP
#define MODL_DENSE_HPP

namespace modl
{

const std::string weight_key = "weight";

const std::string bias_key = "bias";

struct DenseBuilder final : public iLayerBuilder
{
	DenseBuilder (std::string label) : label_(label) {}

	void set_tensor (ade::TensptrT tens) override
	{
		auto node = ead::NodeConverters<PybindT>::to_node(tens);
		if (auto var = std::dynamic_pointer_cast<ead::VariableNode<PybindT>>(node))
		{
			if (var->get_label() == weight_key)
			{
				weight_ = var;
				return;
			}
			else if (var->get_label() == bias_key)
			{
				bias_ = var;
				return;
			}
		}
		logs::warnf("attempt to create dense layer with unknown tensor %s",
			tens->to_string().c_str());
	}

	void set_sublayer (LayerptrT layer) override {} // dense has no sublayer

	LayerptrT build (void) const override;

private:
	ead::VarptrT<PybindT> weight_;

	ead::VarptrT<PybindT> bias_;

	std::string label_;
};

const std::string dense_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "dense",
[](ade::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, dense_layer_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<DenseBuilder>(label);
});

struct Dense final : public iLayer
{
	Dense (ade::DimT nunits, ade::DimT indim,
		eqns::InitF<PybindT> weight_init,
		eqns::InitF<PybindT> bias_init,
		std::string label) :
		iLayer(label),
		weight_(std::make_shared<MarshalVar>(
			weight_init(ade::Shape({nunits, indim}), weight_key)))
	{
		tag(weight_->var_->get_tensor());
		if (bias_init)
		{
			bias_ = std::make_shared<MarshalVar>(
				bias_init(ade::Shape({nunits}), bias_key));
			tag(bias_->var_->get_tensor());
		}
	}

	Dense (ead::VarptrT<PybindT> weight, ead::VarptrT<PybindT> bias,
		std::string label) :
		iLayer(label),
		weight_(std::make_shared<MarshalVar>(weight)),
		bias_(std::make_shared<MarshalVar>(bias))
	{
		tag(weight_->var_->get_tensor());
		if (bias)
		{
			tag(bias_->var_->get_tensor());
		}
	}

	Dense (const Dense& other) :
		iLayer(other)
	{
		copy_helper(other);
	}

	Dense& operator = (const Dense& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	Dense (Dense&& other) = default;

	Dense& operator = (Dense&& other) = default;

	Dense* clone (void) const
	{
		return static_cast<Dense*>(this->clone_impl());
	}

	std::string get_ltype (void) const override
	{
		return dense_layer_key;
	}

	ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> input) const override
	{
		auto out = tenncor::nn::fully_connect({input},
			{ead::convert_to_node(weight_->var_)},
			ead::convert_to_node(bias_->var_));
		recursive_tag(out->get_tensor(), {
			input->get_tensor().get(),
			weight_->var_->get_tensor().get(),
			bias_->var_->get_tensor().get(),
		});
		return out;
	}

	ade::TensT get_contents (void) const override
	{
		return {weight_->var_->get_tensor(), bias_->var_->get_tensor()};
	}

	MarsarrT get_subs (void) const override
	{
		return MarsarrT{weight_, bias_};
	}

private:
	iMarshaler* clone_impl (void) const override
	{
		return new Dense(*this);
	}

	void copy_helper (const Dense& other)
	{
		weight_ = std::make_shared<MarshalVar>(*other.weight_);
		bias_ = std::make_shared<MarshalVar>(*other.bias_);
		tag(weight_->var_->get_tensor());
		if (bias_->var_)
		{
			tag(bias_->var_->get_tensor());
		}
	}

	MarVarsptrT weight_;

	MarVarsptrT bias_;
};

}

#endif // MODL_DENSE_HPP
