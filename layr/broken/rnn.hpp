#include "eteq/generated/api.hpp"

#ifndef MODL_RNN_HPP
#define MODL_RNN_HPP

namespace modl
{

struct RNN final : public iMarshalSet
{
	RNN (teq::DimT n_input, teq::DimT n_output, size_t timestep,
		NonLinearF nonlin, std::string label) :
		iMarshalSet(label), nonlin_(nonlin),
		bias_(eteq::make_variable_scalar<PybindT>(
			0., teq::Shape({n_output}), "bias")
	{
		assert(timestep > 0);
		{
			PybindT bound = 1. / std::sqrt(n_input);
			std::uniform_real_distribution<PybindT> dist(-bound, bound);
			auto gen = [&dist]()
			{
				return dist(eteq::get_engine());
			};
			std::vector<PybindT> wdata(n_output * n_input);
			std::generate(wdata.begin(), wdata.end(), gen);

			eteq::VarptrT<PybindT> weight = eteq::make_variable<PybindT>(
				wdata.data(), teq::Shape({n_output, n_input}), "weight_0");
			layers_.push_back(std::make_shared<MarshalVar>(weight));
		}
		for (size_t i = 1; i < timestep; ++i)
		{
			teq::Shape weight_shape({n_output, n_output});
			teq::NElemT nweight = weight_shape.n_elems();

			PybindT bound = 1. / std::sqrt(n_output);
			std::uniform_real_distribution<PybindT> dist(-bound, bound);
			auto gen = [&dist]()
			{
				return dist(eteq::get_engine());
			};
			std::vector<PybindT> wdata(nweight);
			std::generate(wdata.begin(), wdata.end(), gen);

			eteq::VarptrT<PybindT> weight = eteq::make_variable<PybindT>(
				wdata.data(), weight_shape, fmts::sprintf("weight_%d", i));

			layers_.push_back(std::make_shared<MarshalVar>(weight));
		}
	}

	RNN (const RNN& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	RNN& operator = (const RNN& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	RNN (RNN&& other) = default;

	RNN& operator = (RNN&& other) = default;


	// expect all inputs of shape <n_input, n_batch>
	eteq::NodesT<PybindT> operator () (eteq::NodesT<PybindT> inputs)
	{
		// sanity check
		const teq::Shape& in_shape = input->shape();
		uint8_t ninput = get_ninput();
		if (in_shape.at(0) != ninput)
		{
			logs::fatalf("cannot dbn with input shape %s against n_input %d",
				in_shape.to_string().c_str(), ninput);
		}

		size_t nins = inputs.size();
		if (weights_.size() != nins)
		{
			logs::fatalf("cannot connect %d inputs with %d weights",
				nins, weights_.size());
		}

		eteq::NodesT<PybindT> outs;
		outs.reserve(nins);
		outs.push_back(nonlin_(tenncor::nn::fully_connect(
			{inputs[0]}, {weights_[0]}, bias_)));
		for (uint8_t i = 1; i < ninput; ++i)
		{
			outs.push_back(nonlin(tenncor::nn::fully_connect(
				{outs.back(), inputs[i]},
				{weights_[i - 1], weights_[i]}, bias_)));
		}

		return outs;
	}

	teq::DimT get_ninput (void) const
	{
		return weights_.front()->var_->shape().at(1);
	}

	teq::DimT get_noutput (void) const
	{
		return weights_.back()->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out = weights_;
		out.push_back(bias_);
		return out;
	}

	MarsarrT weights_;

	MarVarsptrT bias_;

	NonLinearF nonlin_;

private:
	void copy_helper (const RNN& other)
	{
		weights_.clear();
		for (const auto& weight : other.weights_)
		{
			weights_.push_back(
				std::make_shared<MarshalVar>(*weight));
		}
		bias_ = std::make_shared<MarshalVar>(*other.bias_);
		nonlin_ = other.nonlin_;
	}

	iMarshaler* clone_impl (void) const override
	{
		return new RNN(*this);
	}
};

}

#endif // MODL_RNN_HPP
