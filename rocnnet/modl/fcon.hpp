#include "ead/generated/api.hpp"
#include "ead/grader.hpp"

#include "rocnnet/eqns/helper.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_FCON_HPP
#define MODL_FCON_HPP

namespace modl
{

const std::string weight_fmt = "weight_%d";
const std::string bias_fmt = "bias_%d";

struct FCon final : public iMarshalSet
{
	FCon (std::vector<uint8_t> n_inputs, uint8_t n_output,
		std::string label) : iMarshalSet(label)
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

			double bound = 1.0 / std::sqrt(n_inputs[i]);
			std::uniform_real_distribution<double> dist(-bound, bound);
			auto gen = [&dist]()
			{
				return dist(ead::get_engine());
			};
			std::vector<double> data(ndata);
			std::generate(data.begin(), data.end(), gen);

			ead::VarptrT<double> weight = ead::make_variable<double>(
				data.data(), shape, fmts::sprintf(weight_fmt, i));
			ead::VarptrT<double> bias = ead::make_variable_scalar<double>(
				0.0, ade::Shape({n_output}), fmts::sprintf(bias_fmt, i));
			weight_bias_.push_back({
				std::make_shared<MarshalVar>(weight),
				std::make_shared<MarshalVar>(bias),
			});
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


	ead::NodeptrT<double> operator () (ead::NodesT<double> inputs)
	{
		size_t n = inputs.size();
		if (n != weight_bias_.size())
		{
			logs::fatalf("number of inputs must be exactly %d", n);
		}
		ead::NodeptrT<double> out = eqns::weighed_bias_add(
			age::matmul(inputs[0],
				ead::convert_to_node(weight_bias_[0].weight_->var_)),
			ead::convert_to_node(weight_bias_[0].bias_->var_));
		for (size_t i = 1; i < n; ++i)
		{
			out = eqns::weighed_bias_add(
				age::matmul(inputs[i],
					ead::convert_to_node(weight_bias_[i].weight_->var_)),
				ead::convert_to_node(weight_bias_[i].bias_->var_));
		}
		return out;
	}

	uint8_t get_ninput (void) const
	{
		return weight_bias_[0].weight_->var_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weight_bias_[0].weight_->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out;
		for (const BiasedLayer& wbpair : weight_bias_)
		{
			out.push_back(wbpair.weight_);
			out.push_back(wbpair.bias_);
		}
		return out;
	}

private:
	void copy_helper (const FCon& other)
	{
		weight_bias_.clear();
		for (const BiasedLayer& opair : other.weight_bias_)
		{
			weight_bias_.push_back({
				std::make_shared<MarshalVar>(*opair.weight_),
				std::make_shared<MarshalVar>(*opair.bias_),
			});
		}
	}

	iMarshaler* clone_impl (void) const override
	{
		return new FCon(*this);
	}

	struct BiasedLayer
	{
		MarVarsptrT weight_;
		MarVarsptrT bias_;
	};

	std::vector<BiasedLayer> weight_bias_;
};

using FConptrT = std::shared_ptr<FCon>;

}

#endif // MODL_FCON_HPP
