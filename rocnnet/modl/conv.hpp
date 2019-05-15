#include "prx/api.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_CONV_HPP
#define MODL_CONV_HPP

namespace modl
{

struct Conv final : public iMarshalSet
{
	Conv (std::pair<ade::DimT,ade::DimT> filter_hw, ade::DimT in_ncol,
		uinade::DimT out_ncol, std::string label) : iMarshalSet(label)
	{
		ade::Shape kernelshape({out_ncol, in_ncol,
			filter_hw.second, filter_hw.first});
		size_t ndata = kernelshape.n_elems();

		size_t input_size = filter_hw.first * filter_hw.second * in_ncol;
		PybindT bound = 1.0 / std::sqrt(input_size);
		std::uniform_real_distribution<PybindT> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(ead::get_engine());
		};
		std::vector<PybindT> data(ndata);
		std::generate(data.begin(), data.end(), gen);

		ead::VarptrT<PybindT> weight = ead::make_variable<PybindT>(
			data.data(), kernelshape, "weight");
		ead::VarptrT<PybindT> bias = ead::make_variable_scalar<PybindT>(
			0.0, ade::Shape({out_ncol}), "bias");
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

	ead::NodeptrT<PybindT> operator () (ead::NodeptrT<PybindT> input)
	{
		return prx::conv2d(input,
			ead::convert_to_node<PybindT>(weight_->var_),
			ead::convert_to_node<PybindT>(bias_->var_));
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
