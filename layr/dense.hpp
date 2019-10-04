///
/// dense.hpp
/// layr
///
/// Purpose:
/// Implement fully connected layer
///

#include "eteq/generated/api.hpp"

#include "layr/init.hpp"
#include "layr/layer.hpp"

#ifndef LAYR_DENSE_HPP
#define LAYR_DENSE_HPP

namespace layr
{

/// Fully connected weight label
const std::string dense_weight_key = "weight";

/// Fully connected bias label
const std::string dense_bias_key = "bias";

/// Builder implementation for fully connected layer
struct DenseBuilder final : public iLayerBuilder
{
	DenseBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override
	{
		if (target == dense_weight_key)
		{
			weight_ = eteq::NodeConverters<PybindT>::to_node(tens);
			return;
		}
		else if (target == dense_bias_key)
		{
			bias_ = eteq::NodeConverters<PybindT>::to_node(tens);
			return;
		}
		logs::warnf("attempt to create dense layer "
			"with unknown tensor `%s` with label `%s`",
			tens->to_string().c_str(), target.c_str());
	}

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override {} // dense has no sublayer

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	NodeptrT weight_ = nullptr;

	NodeptrT bias_ = nullptr;

	std::string label_;
};

/// Identifier for fully connected layer
const std::string dense_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "dense",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<DenseBuilder>(label);
});

/// Layer implementation to apply fully_connect functions to weight and optional bias
struct Dense final : public iLayer
{
	Dense (teq::DimT nunits, teq::DimT indim,
		layr::InitF<PybindT> weight_init,
		layr::InitF<PybindT> bias_init,
		const std::string& label) :
		label_(label),
		weight_(weight_init(teq::Shape({nunits, indim}), dense_weight_key))
	{
		tag(weight_->get_tensor(), LayerId(dense_weight_key));
		if (bias_init)
		{
			bias_ = bias_init(teq::Shape({nunits}), dense_bias_key);
			tag(bias_->get_tensor(), LayerId(dense_bias_key));
		}
	}

	Dense (NodeptrT weight, NodeptrT bias, std::string label) :
		label_(label),
		weight_(weight),
		bias_(bias)
	{
		tag(weight_->get_tensor(), LayerId(dense_weight_key));
		if (bias)
		{
			tag(bias_->get_tensor(), LayerId(dense_bias_key));
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

	/// Return deep copy of this layer with prefixed label
	Dense* clone (std::string label_prefix = "") const
	{
		return static_cast<Dense*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	size_t get_ninput (void) const override
	{
		return weight_->shape().at(1);
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		return weight_->shape().at(0);
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return dense_layer_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		return {
			weight_->get_tensor(),
			nullptr == bias_ ? nullptr : bias_->get_tensor(),
		};
	}

	/// Implementation of iLayer
	NodeptrT connect (NodeptrT input) const override
	{
		auto out = tenncor::nn::fully_connect({input}, {weight_}, bias_);
		teq::TensSetT leaves = {
			input->get_tensor().get(),
			weight_->get_tensor().get(),
		};
		if (bias_)
		{
			leaves.emplace(bias_->get_tensor().get());
		}
		recursive_tag(out->get_tensor(), leaves, LayerId());
		return out;
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new Dense(*this, label_prefix);
	}

	void copy_helper (const Dense& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		weight_ = NodeptrT(other.weight_->clone());
		tag(weight_->get_tensor(), LayerId(dense_weight_key));
		if (other.bias_)
		{
			bias_ = NodeptrT(other.bias_->clone());
			tag(bias_->get_tensor(), LayerId(dense_bias_key));
		}
	}

	std::string label_;

	NodeptrT weight_;

	NodeptrT bias_;
};

/// Smart pointer of fully connected layer
using DenseptrT = std::shared_ptr<Dense>;

}

#endif // LAYR_DENSE_HPP
