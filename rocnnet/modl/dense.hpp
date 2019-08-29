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

	void set_tensor (ade::TensptrT tens, std::string target) override
	{
		if (target == weight_key)
		{
			weight_ = ead::NodeConverters<PybindT>::to_node(tens);
			return;
		}
		else if (target == bias_key)
		{
			bias_ = ead::NodeConverters<PybindT>::to_node(tens);
			return;
		}
		logs::warnf("attempt to create dense layer "
			"with unknown tensor `%s` with label `%s`",
			tens->to_string().c_str(), target.c_str());
	}

	void set_sublayer (LayerptrT layer) override {} // dense has no sublayer

	LayerptrT build (void) const override;

private:
	ead::NodeptrT<PybindT> weight_ = nullptr;

	ead::NodeptrT<PybindT> bias_ = nullptr;

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
		const std::string& label) :
		label_(label),
		weight_(weight_init(ade::Shape({nunits, indim}), weight_key))
	{
		tag(weight_->get_tensor(), LayerId(weight_key));
		if (bias_init)
		{
			bias_ = bias_init(ade::Shape({nunits}), bias_key);
			tag(bias_->get_tensor(), LayerId(bias_key));
		}
	}

	Dense (ead::NodeptrT<PybindT> weight,
		ead::NodeptrT<PybindT> bias,
		std::string label) :
		label_(label),
		weight_(weight),
		bias_(bias)
	{
		tag(weight_->get_tensor(), LayerId(weight_key));
		if (bias)
		{
			tag(bias_->get_tensor(), LayerId(bias_key));
		}
	}

	Dense (const Dense& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
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

	Dense* clone (std::string label_prefix = "") const
	{
		return static_cast<Dense*>(this->clone_impl(label_prefix));
	}

	size_t get_ninput (void) const override
	{
		return weight_->shape().at(1);
	}

	size_t get_noutput (void) const override
	{
		return weight_->shape().at(0);
	}

	std::string get_ltype (void) const override
	{
		return dense_layer_key;
	}

	std::string get_label (void) const override
	{
		return label_;
	}

	ead::NodeptrT<PybindT> connect (ead::NodeptrT<PybindT> input) const override
	{
		auto out = tenncor::nn::fully_connect({input}, {weight_}, bias_);
		recursive_tag(out->get_tensor(), {
			input->get_tensor().get(),
			weight_->get_tensor().get(),
			bias_->get_tensor().get(),
		}, LayerId());
		return out;
	}

	ade::TensT get_contents (void) const override
	{
		return {
			weight_->get_tensor(),
			nullptr == bias_ ? nullptr : bias_->get_tensor(),
		};
	}

private:
	iLayer* clone_impl (std::string label_prefix) const override
	{
		return new Dense(*this, label_prefix);
	}

	void copy_helper (const Dense& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		weight_ = std::make_shared<ead::VariableNode<PybindT>>(
			std::shared_ptr<ead::Variable<PybindT>>(
				ead::Variable<PybindT>::get(
					*static_cast<ead::Variable<PybindT>*>(
						other.weight_->get_tensor().get()))));
		tag(weight_->get_tensor(), LayerId(weight_key));
		if (other.bias_)
		{
			bias_ = std::make_shared<ead::VariableNode<PybindT>>(
				std::shared_ptr<ead::Variable<PybindT>>(
					ead::Variable<PybindT>::get(
						*static_cast<ead::Variable<PybindT>*>(
							other.bias_->get_tensor().get()))));
			tag(bias_->get_tensor(), LayerId(bias_key));
		}
	}

	std::string label_;

	ead::NodeptrT<PybindT> weight_;

	ead::NodeptrT<PybindT> bias_;
};

using DenseptrT = std::shared_ptr<Dense>;

}

#endif // MODL_DENSE_HPP
