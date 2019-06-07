#include "ead/generated/api.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_RBM_HPP
#define MODL_RBM_HPP

namespace modl
{

struct RBM final : public iMarshalSet
{
	RBM (ade::DimT n_input,
		std::vector<ade::DimT> layer_outs, std::string label) :
		iMarshalSet(label)
	{
		if (layer_outs.empty())
		{
			logs::fatal("cannot create RBM with no layers specified");
		}

		for (size_t i = 0, n = layer_outs.size(); i < n; ++i)
		{
			ade::DimT n_output = layer_outs[i];
			ade::Shape weight_shape({n_output, n_input});
			ade::NElemT nweight = weight_shape.n_elems();

			PybindT bound = 4 * std::sqrt(6.0 / (n_output + n_input));
			std::uniform_real_distribution<PybindT> dist(-bound, bound);
			auto gen = [&dist]()
			{
				return dist(ead::get_engine());
			};
			std::vector<PybindT> wdata(nweight);
			std::generate(wdata.begin(), wdata.end(), gen);

			ead::VarptrT<PybindT> weight = ead::make_variable<PybindT>(
				wdata.data(), weight_shape, fmts::sprintf("weight_%d", i));

			ead::VarptrT<PybindT> hbias = ead::make_variable_scalar<PybindT>(
				0.0, ade::Shape({n_output}), fmts::sprintf("hbias_%d", i));

			ead::VarptrT<PybindT> vbias = ead::make_variable_scalar<PybindT>(
				0.0, ade::Shape({n_input}), fmts::sprintf("vbias_%d", i));

			layers_.push_back(HiddenLayer{
				std::make_shared<MarshalVar>(weight),
				std::make_shared<MarshalVar>(hbias),
				std::make_shared<MarshalVar>(vbias),
			});
			n_input = n_output;
		}
	}

	RBM (const RBM& other) : iMarshalSet(other)
	{
		copy_helper(other);
	}

	RBM& operator = (const RBM& other)
	{
		if (this != &other)
		{
			iMarshalSet::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	RBM (RBM&& other) = default;

	RBM& operator = (RBM&& other) = default;


	// input of shape <n_input, n_batch>
	// propagate upwards (towards visibleness)
	ead::NodeptrT<PybindT> operator () (ead::NodeptrT<PybindT> input,
		NonLinearsT nonlinearities)
	{
		// sanity check
		const ade::Shape& in_shape = input->shape();
		uint8_t ninput = get_ninput();
		if (in_shape.at(0) != ninput)
		{
			logs::fatalf(
				"cannot generate rbm with input shape %s against n_input %d",
				in_shape.to_string().c_str(), ninput);
		}

		size_t nlayers = layers_.size();
		size_t nnlin = nonlinearities.size();
		if (nnlin != nlayers)
		{
			logs::fatalf(
				"cannot generate rbm of %d layers with %d nonlinearities",
				nlayers, nnlin);
		}

		ead::NodeptrT<PybindT> out = input;
		for (size_t i = 0; i < nlayers; ++i)
		{
			// weight is <n_hidden, n_input>
			// in is <n_input, ?>
			// out = in @ weight, so out is <n_hidden, ?>
			auto hypothesis = age::fully_connect({out},
				{ead::convert_to_node(layers_[i].weight_->var_)},
				ead::convert_to_node(layers_[i].hbias_->var_));
			out = nonlinearities[i](hypothesis);
		}
		return out;
	}

	// input of shape <n_hidden, n_batch>
	ead::NodeptrT<PybindT> prop_down (ead::NodeptrT<PybindT> hidden,
		NonLinearsT nonlinearities)
	{
		// sanity check
		const ade::Shape& out_shape = hidden->shape();
		uint8_t noutput = get_noutput();
		if (out_shape.at(0) != noutput)
		{
			logs::fatalf(
				"cannot prop down rbm with output shape %s against n_output %d",
				out_shape.to_string().c_str(), noutput);
		}

		size_t nlayers = layers_.size();
		size_t nnlin = nonlinearities.size();
		if (nnlin != nlayers)
		{
			logs::fatalf(
				"cannot generate rbm of %d layers with %d nonlinearities",
				nlayers, nnlin);
		}

		ead::NodeptrT<PybindT> out = hidden;
		for (size_t i = 0; i < nlayers; ++i)
		{
			size_t index = nlayers - i - 1;
			// weight is <n_hidden, n_input>
			// in is <n_hidden, ?>
			// out = in @ weight.T, so out is <n_input, ?>
			auto hypothesis = age::fully_connect({out},
				{age::transpose(
					ead::convert_to_node(layers_[index].weight_->var_))},
				ead::convert_to_node(layers_[index].vbias_->var_));
			out = nonlinearities[index](hypothesis);
		}
		return out;
	}

	uint8_t get_ninput (void) const
	{
		return layers_.front().weight_->var_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return layers_.back().weight_->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		MarsarrT out;
		out.reserve(3 * layers_.size());
		for (const HiddenLayer& layer : layers_)
		{
			out.push_back(layer.weight_);
			out.push_back(layer.hbias_);
			out.push_back(layer.vbias_);
		}
		return out;
	}

	struct HiddenLayer
	{
		MarVarsptrT weight_;

		MarVarsptrT hbias_;

		MarVarsptrT vbias_;
	};

	std::vector<HiddenLayer> layers_;

private:
	void copy_helper (const RBM& other)
	{
		layers_.clear();
		for (const HiddenLayer& olayer : other.layers_)
		{
			layers_.push_back(HiddenLayer{
				std::make_shared<MarshalVar>(*olayer.weight_),
				std::make_shared<MarshalVar>(*olayer.hbias_),
				std::make_shared<MarshalVar>(*olayer.vbias_),
			});
		}
	}

	iMarshaler* clone_impl (void) const override
	{
		return new RBM(*this);
	}
};

using RBMptrT = std::shared_ptr<RBM>;

}

#endif // MODL_RBM_HPP
