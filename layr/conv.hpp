///
/// conv.hpp
/// layr
///
/// Purpose:
/// Implement convolutional layer
///

#include "teq/ilayer.hpp"

#include "eteq/generated/api.hpp"

#include "layr/init.hpp"

#ifndef LAYR_CONV_HPP
#define LAYR_CONV_HPP

namespace layr
{

/// Convolutional weight label
const std::string conv_weight_key = "weight";

/// Convolutional bias label
const std::string conv_bias_key = "bias";

/// Layer implementation to apply conv2d functions to weight and optional bias
template <typename T=PybindT>
struct Conv final : public teq::iLayer
{
	Conv (std::pair<teq::DimT,teq::DimT> filter_hw,
		teq::DimT in_ncol, teq::DimT out_ncol,
		const std::string& label,
		std::pair<teq::DimT,teq::DimT> zero_padding = {0, 0}) :
		iLayer(teq::ShapeSignature({in_ncol, 0, 0, 0}))
		label_(label),
		weight_(unif_xavier_init<T>(1)(teq::Shape({
			out_ncol, in_ncol, filter_hw.second, filter_hw.first}),
			conv_weight_key)),
		bias_(zero_init<T>()(teq::Shape({out_ncol}), conv_bias_key)),
		zero_padding_(zero_padding)
	{
		output_ = connect(this->input_);
	}

	Conv (const Conv& other,
		std::string label_prefix = "") :
		iLayer(other)
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
	teq::ShapeSignature get_output_sign (void) const override
	{
		return output_->shape_sign();
	}

	/// Implementation of iLayer
	teq::TensptrT get_output (void) const override
	{
		return output_->get_tensor();
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return "conv";
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_storage (void) const override
	{
		return {weight_, bias_};
	}

	/// Implementation of iLayer
	teq::TensptrT connect (teq::TensptrT input) const override
	{
		auto output = tenncor::nn::conv2d(eteq::to_link<T>(input),
			eteq::to_link<PybindT>(weight_), zero_padding_);
		if (bias_)
		{
			output = output + tenncor::best_extend(
				eteq::to_link<PybindT>(bias_), output->shape_sign());
		}
		return output;
	}

	std::pair<teq::DimT,teq::DimT> get_padding (void) const
	{
		return zero_padding_;
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
		if (other.bias_)
		{
			bias_ = teq::TensptrT(other.bias_->clone());
		}
		zero_padding_ = other.zero_padding_;
		output_ = connect(this->input_);
	}

	std::string label_;

	teq::TensptrT weight_;

	teq::TensptrT bias_ = nullptr;

	std::pair<teq::DimT,teq::DimT> zero_padding_;

	eteq::LinkptrT<T> output_;
};

/// Smart pointer of convolutional layer
using ConvptrT = std::shared_ptr<Conv>;

}

#endif // LAYR_CONV_HPP
