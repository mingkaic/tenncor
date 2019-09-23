#include "eteq/generated/api.hpp"

#include "layr/layer.hpp"

#ifndef LAYR_CONV_HPP
#define LAYR_CONV_HPP

namespace layr
{

const std::string conv_weight_key = "weight";

const std::string conv_bias_key = "bias";

struct ConvBuilder final : public iLayerBuilder
{
	ConvBuilder (std::string label) : label_(label) {}

	void set_tensor (teq::TensptrT tens, std::string target) override
	{
		if (target == conv_weight_key)
		{
			weight_ = eteq::NodeConverters<PybindT>::to_node(tens);
			return;
		}
		else if (target == conv_bias_key)
		{
			bias_ = eteq::NodeConverters<PybindT>::to_node(tens);
			return;
		}
		logs::warnf("attempt to create convolution layer "
			"with unknown tensor `%s` with label `%s`",
			tens->to_string().c_str(), target.c_str());
	}

	void set_sublayer (LayerptrT layer) override {} // dense has no sublayer

	LayerptrT build (void) const override;

private:
	eteq::NodeptrT<PybindT> weight_ = nullptr;

	eteq::NodeptrT<PybindT> bias_ = nullptr;

	std::string label_;
};

const std::string conv_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "conv",
[](teq::TensrefT ref, std::string label)
{
	get_layer_reg().layer_tag(ref, conv_layer_key, label);
},
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<ConvBuilder>(label);
});

struct Conv final : public iLayer
{
	Conv (std::pair<teq::DimT,teq::DimT> filter_hw,
		teq::DimT in_ncol, teq::DimT out_ncol,
		const std::string& label) :
		label_(label)
	{
		teq::Shape kernelshape({out_ncol, in_ncol,
			filter_hw.second, filter_hw.first});
		size_t ndata = kernelshape.n_elems();

		size_t input_size = filter_hw.first * filter_hw.second * in_ncol;
		PybindT bound = 1.0 / std::sqrt(input_size);
		std::uniform_real_distribution<PybindT> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(eteq::get_engine());
		};
		std::vector<PybindT> data(ndata);
		std::generate(data.begin(), data.end(), gen);

		eteq::VarptrT<PybindT> weight = eteq::make_variable<PybindT>(
			data.data(), kernelshape, "weight");
		eteq::VarptrT<PybindT> bias = eteq::make_variable_scalar<PybindT>(
			0.0, teq::Shape({out_ncol}), "bias");
		weight_ = std::make_shared<MarshalVar>(weight);
		bias_ = std::make_shared<MarshalVar>(bias);
	}

	Conv (eteq::NodeptrT<PybindT> weight,
		eteq::NodeptrT<PybindT> bias,
		std::string label) :
		label_(label),
		weight_(weight),
		bias_(bias)
	{
		tag(weight_->get_tensor(), LayerId(conv_weight_key));
		if (bias)
		{
			tag(bias_->get_tensor(), LayerId(conv_bias_key));
		}
	}

	Conv (const Conv& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	Conv& operator = (const Conv& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	Conv (Conv&& other) = default;

	Conv& operator = (Conv&& other) = default;

	Conv* clone (std::string label_prefix = "") const
	{
		return static_cast<Conv*>(this->clone_impl(label_prefix));
	}

	uint8_t get_ninput (void) const override
	{
		return weight_->shape().at(1);
	}

	uint8_t get_noutput (void) const override
	{
		return weight_->shape().at(0);
	}

	std::string get_ltype (void) const override
	{
		return conv_layer_key;
	}

	std::string get_label (void) const override
	{
		return label_;
	}

	eteq::NodeptrT<PybindT> connect (eteq::NodeptrT<PybindT> input) const override
	{
		auto out = tenncor::nn::conv2d(input,
			eteq::convert_to_node<PybindT>(weight_),
			eteq::convert_to_node<PybindT>(bias_));
		auto leaves = {
			input->get_tensor().get(),
			weight_->get_tensor().get(),
		};
		if (bias)
		{
			leaves.push_back(bias_->get_tensor().get());
		}
		recursive_tag(out->get_tensor(), leaves, LayerId());
		return out;
	}

	teq::TensT get_contents (void) const override
	{
		return {
			weight_->get_tensor(),
			nullptr == bias_ ? nullptr : bias_->get_tensor(),
		};
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new Conv(*this, label_prefix);
	}

	void copy_helper (const Conv& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		weight_ = std::make_shared<eteq::VariableNode<PybindT>>(
			std::shared_ptr<eteq::Variable<PybindT>>(
				eteq::Variable<PybindT>::get(
					*static_cast<eteq::Variable<PybindT>*>(
						other.weight_->get_tensor().get()))));
		tag(weight_->get_tensor(), LayerId(conv_weight_key));
		if (other.bias_)
		{
			bias_ = std::make_shared<eteq::VariableNode<PybindT>>(
				std::shared_ptr<eteq::Variable<PybindT>>(
					eteq::Variable<PybindT>::get(
						*static_cast<eteq::Variable<PybindT>*>(
							other.bias_->get_tensor().get()))));
			tag(bias_->get_tensor(), LayerId(conv_bias_key));
		}
	}

	std::string label_;

	eteq::NodeptrT<PybindT> weight_;

	eteq::NodeptrT<PybindT> bias_ = nullptr;
};

using ConvptrT = std::shared_ptr<Conv>;

}

#endif // LAYR_CONV_HPP
