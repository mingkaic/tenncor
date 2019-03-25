#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/operator.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_CONV_HPP
#define MODL_CONV_HPP

namespace modl
{

struct Conv final : public iMarshalSet
{
	Conv (std::pair<uint8_t,uint8_t> filter_hw, uint8_t in_ncol,
		uint8_t out_ncol, std::string label) : iMarshalSet(label)
	{
		ade::Shape shape({filter_hw.first,
			filter_hw.second, in_ncol, out_ncol});
		size_t ndata = shape.n_elems();

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
			data.data(), shape, "weight");
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
		return age::add(age::convolution(input,
			ead::convert_to_node<PybindT>(weight_->var_)),
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

	MarVarsptrT weight_;

	MarVarsptrT bias_;
};

}

#endif // MODL_CONV_HPP
