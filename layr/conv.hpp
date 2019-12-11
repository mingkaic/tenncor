///
/// conv.hpp
/// layr
///
/// Purpose:
/// Implement convolutional layer
///

#include "eteq/generated/api.hpp"

#include "layr/layer.hpp"

#ifndef LAYR_CONV_HPP
#define LAYR_CONV_HPP

namespace layr
{

#ifdef FULLTYPE
using DArgT = teq::DimT;
#else
using DArgT = PybindT;
#endif

/// Convolutional weight label
const std::string conv_weight_key = "weight";

/// Convolutional bias label
const std::string conv_bias_key = "bias";

/// Argument when convolving label
const std::string conv_arg_key = "args";

/// Builder implementation for convolution layer
struct ConvBuilder final : public iLayerBuilder
{
	ConvBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override
	{
		if (target == conv_weight_key)
		{
			weight_ = tens;
			return;
		}
		else if (target == conv_bias_key)
		{
			bias_ = tens;
			return;
		}
		else if (target == conv_arg_key)
		{
			arg_ = eteq::to_link<DArgT>(tens);
		}
		logs::warnf("attempt to create convolution layer "
			"with unknown tensor `%s` of label `%s`",
			tens->to_string().c_str(), target.c_str());
	}

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override {} // dense has no sublayer

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	teq::TensptrT weight_ = nullptr;

	teq::TensptrT bias_ = nullptr;

	eteq::LinkptrT<DArgT> arg_ = nullptr;

	std::string label_;
};

/// Identifier for convolutional layer
const std::string conv_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "conv",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<ConvBuilder>(label);
});

/// Layer implementation to apply conv2d functions to weight and optional bias
struct Conv final : public iLayer
{
	Conv (std::pair<teq::DimT,teq::DimT> filter_hw,
		teq::DimT in_ncol, teq::DimT out_ncol,
		const std::string& label,
		std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0}) :
		label_(label)
	{
		teq::Shape kernelshape({out_ncol, in_ncol,
			filter_hw.second, filter_hw.first});
		size_t ndata = kernelshape.n_elems();

		size_t input_size = filter_hw.first * filter_hw.second * in_ncol;
		PybindT bound = 1. / std::sqrt(input_size);
		std::uniform_real_distribution<PybindT> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(eigen::get_engine());
		};
		std::vector<PybindT> data(ndata);
		std::generate(data.begin(), data.end(), gen);

		weight_ = eteq::make_variable<PybindT>(
			data.data(), kernelshape, conv_weight_key);
		bias_ = eteq::make_variable_scalar<PybindT>(
			0., teq::Shape({out_ncol}), conv_bias_key);
		tag(weight_, LayerId(conv_weight_key));
		tag(bias_, LayerId(conv_bias_key));
		if (zero_padding.first > 0 || zero_padding.second > 0)
		{
			std::vector<DArgT> buffer = {
				(DArgT) zero_padding.first, (DArgT) zero_padding.second};
			arg_ = eteq::make_constant<DArgT>(
				buffer.data(), teq::Shape({2}));
			tag(arg_->get_tensor(), LayerId(conv_arg_key));
		}

		placeholder_connect();
	}

	Conv (teq::TensptrT weight, teq::TensptrT bias,
		eteq::LinkptrT<DArgT> arg, const std::string& label) :
		label_(label),
		weight_(weight),
		bias_(bias),
		arg_(arg)
	{
		tag(weight_, LayerId(conv_weight_key));
		if (bias)
		{
			tag(bias_, LayerId(conv_bias_key));
		}
		if (arg)
		{
			tag(arg_->get_tensor(), LayerId(conv_arg_key));
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

	/// Return deep copy of this layer with prefixed label
	Conv* clone (std::string label_prefix = "") const
	{
		return static_cast<Conv*>(this->clone_impl(label_prefix));
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
	teq::ShapeSignature get_input_sign (void) const override
	{
		// weight has shape [out, in, wwidth, wheight]
		// so input must have shape [in, *, *, *]
		teq::Shape wshape = weight_->shape();
		return teq::ShapeSignature(
			std::vector<teq::DimT>{wshape.at(1)});
	}

	/// Implementation of iLayer
	teq::ShapeSignature get_output_sign (void) const override
	{
		// weight has shape [out, in, width, height]
		// so output must have shape [out, *, *, *]
		teq::Shape wshape = weight_->shape();
		return teq::ShapeSignature(
			std::vector<teq::DimT>{wshape.at(0)});
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return conv_layer_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		return {weight_, bias_};
	}

	/// Implementation of iLayer
	LinkptrT connect (LinkptrT input) const override
	{
		auto output = tenncor::nn::conv2d(input,
			eteq::to_link<PybindT>(weight_), get_padding());
		if (bias_)
		{
			output = output + tenncor::best_extend(
				eteq::to_link<PybindT>(bias_), output->shape_sign());
		}
		return output;
	}

	std::pair<teq::DimT,teq::DimT> get_padding (void) const
	{
		if (arg_)
		{
			DArgT* adata = arg_->data();
			return {adata[0], adata[1]};
		}
		return {0, 0};
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new Conv(*this, label_prefix);
	}

	void copy_helper (const Conv& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		weight_ = teq::TensptrT(other.weight_->clone());
		tag(weight_, LayerId(conv_weight_key));
		if (other.bias_)
		{
			bias_ = teq::TensptrT(other.bias_->clone());
			tag(bias_, LayerId(conv_bias_key));
		}
		if (other.arg_)
		{
			arg_ = eteq::LinkptrT<DArgT>(other.arg_->clone());
			tag(arg_->get_tensor(), LayerId(conv_arg_key));
		}

		this->input_ = nullptr;
		this->placeholder_connect();
	}

	std::string label_;

	teq::TensptrT weight_;

	teq::TensptrT bias_ = nullptr;

	eteq::LinkptrT<DArgT> arg_ = nullptr;
};

/// Smart pointer of convolutional layer
using ConvptrT = std::shared_ptr<Conv>;

}

#endif // LAYR_CONV_HPP
