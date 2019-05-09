#include "ead/generated/api.hpp"

#include "ead/grader.hpp"

#include "rocnnet/eqns/helper.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_FCON_HPP
#define MODL_FCON_HPP

namespace modl
{

const std::string weight_fmt = "weight_%d";
const std::string bias_fmt = "bias";

struct FCon final : public iMarshalSet
{
	FCon (std::vector<uint8_t> n_inputs, uint8_t n_output,
		std::string label) : iMarshalSet(label),
		bias_(std::make_shared<MarshalVar>(
			ead::make_variable_scalar<PybindT>(
			0.0, ade::Shape({n_output}), bias_fmt)))
	{
		size_t n = n_inputs.size();
		if (n == 0)
		{
			logs::fatal("cannot create FCon with no inputs");
		}
		for (size_t i = 0; i < n; ++i)
		{
			ade::Shape shape({n_output, n_inputs[i]});
			size_t ndata = shape.n_elems();

			PybindT bound = 1.0 / std::sqrt(n_inputs[i]);
			std::uniform_real_distribution<PybindT> dist(-bound, bound);
			auto gen = [&dist]()
			{
				return dist(ead::get_engine());
			};
			std::vector<PybindT> data(ndata);
			std::generate(data.begin(), data.end(), gen);

			ead::VarptrT<PybindT> weight = ead::make_variable<PybindT>(
				data.data(), shape, fmts::sprintf(weight_fmt, i));
			weights_.push_back(std::make_shared<MarshalVar>(weight));
		}
	}

	FCon (const FCon& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	FCon& operator = (const FCon& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	FCon (FCon&& other) = default;

	FCon& operator = (FCon&& other) = default;


	ead::NodeptrT<PybindT> operator () (ead::NodesT<PybindT> inputs)
	{
		size_t n = inputs.size();
		if (n != weights_.size())
		{
			logs::fatalf("number of inputs must be exactly %d", n);
		}
		ead::NodeptrT<PybindT> out = age::matmul(inputs[0],
			ead::convert_to_node(weights_[0]->var_));
		for (size_t i = 1; i < n; ++i)
		{
			out = age::add(age::matmul(inputs[i],
				ead::convert_to_node(weights_[i]->var_)));
		}
		return eqns::weighed_bias_add(out,
			ead::convert_to_node(bias_->var_));
	}

	uint8_t get_ninput (void) const
	{
		return weights_[0]->var_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weights_[0]->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out;
		for (const MarVarsptrT& weight : weights_)
		{
			out.push_back(weight);
		}
		out.push_back(bias_);
		return out;
	}

private:
	void copy_helper (const FCon& other)
	{
		weights_.clear();
		for (const MarVarsptrT& weight : other.weights_)
		{
			weights_.push_back(
				std::make_shared<MarshalVar>(*weight));
		}
		bias_ = std::make_shared<MarshalVar>(*other.bias_),
	}

	iMarshaler* clone_impl (void) const override
	{
		return new FCon(*this);
	}

	std::vector<MarVarsptrT> weights_;

	MarVarsptrT bias_;
};

using FConptrT = std::shared_ptr<FCon>;

}

#endif // MODL_FCON_HPP
