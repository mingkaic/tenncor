#include "eteq/generated/api.hpp"

#include "modl/marshal.hpp"

#ifndef MODL_CONV_HPP
#define MODL_CONV_HPP

namespace modl
{

struct Conv final : public iMarshalSet
{
	Conv (std::pair<teq::DimT,teq::DimT> filter_hw, teq::DimT in_ncol,
		uinade::DimT out_ncol, std::string label) : iMarshalSet(label)
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

	Conv (const Conv& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	Conv& operator = (const Conv& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	Conv (Conv&& other) = default;

	Conv& operator = (Conv&& other) = default;

	eteq::NodeptrT<PybindT> operator () (eteq::NodeptrT<PybindT> input)
	{
		return tenncor::nn::conv2d(input,
			eteq::convert_to_node<PybindT>(weight_->var_),
			eteq::convert_to_node<PybindT>(bias_->var_));
	}

	uint8_t get_ninput (void) const
	{
		return weight_->var_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weight_->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		return {weight_, bias_};
	}

	MarVarsptrT weight_;

	MarVarsptrT bias_;

private:
	void copy_helper (const Conv& other)
	{
		weight_ = std::make_shared<MarshalVar>(*other.weight_);
		bias_ = std::make_shared<MarshalVar>(*other.bias_);
	}

	iMarshaler* clone_impl (void) const override
	{
		return new Conv(*this);
	}
};

}

#endif // MODL_CONV_HPP
