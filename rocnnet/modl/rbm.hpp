#include "prx/api.hpp"

#include "rocnnet/eqns/helper.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_RBM_HPP
#define MODL_RBM_HPP

namespace modl
{

struct RBM final : public iMarshalSet
{
	RBM (ade::DimT n_input, std::vector<LayerInfo> layers, std::string label) :
		iMarshalSet(label)
	{
		if (layers.empty())
		{
			logs::fatal("cannot create RBM with no layers specified");
		}

		for (size_t i = 0, n = layers.size(); i < n; ++i)
		{
			ade::DimT n_output = layers[i].n_out_;
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
				layers[i].hidden_,
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
	ead::NodeptrT<PybindT> operator () (ead::NodeptrT<PybindT> input)
	{
		// sanity check
		const ade::Shape& in_shape = input->shape();
		uint8_t ninput = get_ninput();
		if (in_shape.at(0) != ninput)
		{
			logs::fatalf("cannot dbn with input shape %s against n_input %d",
				in_shape.to_string().c_str(), ninput);
		}

		ead::NodeptrT<PybindT> out = input;
		for (HiddenLayer& layer : layers_)
		{
			// weight is <n_hidden, n_input>
			// in is <n_input, ?>
			// out = in @ weight, so out is <n_hidden, ?>
			auto hypothesis = prx::fully_connect({out},
				{ead::convert_to_node(layer.weight_->var_)},
				ead::convert_to_node(layer.hbias_->var_));
			out = layer.hidden_(hypothesis);
		}
		return out;
	}

	// input of shape <n_hidden, n_batch>
	ead::NodeptrT<PybindT> prop_down (ead::NodeptrT<PybindT> hidden)
	{
		ead::NodeptrT<PybindT> out = hidden;
		for (HiddenLayer& layer : layers_)
		{
			// weight is <n_hidden, n_input>
			// in is <n_hidden, ?>
			// out = in @ weight.T, so out is <n_input, ?>
			auto hypothesis = prx::fully_connect({out},
				{age::transpose(ead::convert_to_node(layer.weight_->var_))},
				ead::convert_to_node(layer.vbias_->var_));
			out = layer.hidden_(hypothesis);
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

		NonLinearF hidden_;
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
				olayer.hidden_
			});
		}
	}

	iMarshaler* clone_impl (void) const override
	{
		return new RBM(*this);
	}
};

using RBMptrT = std::shared_ptr<RBM>;

// recreate input using hidden distribution
// output shape of input->shape()
ead::NodeptrT<PybindT> reconstruct_visible (RBM& rbm, ead::NodeptrT<PybindT> input)
{
	ead::NodeptrT<PybindT> hidden_dist = rbm(input);
	ead::NodeptrT<PybindT> hidden_sample = eqns::one_binom(hidden_dist);
	return rbm.prop_down(hidden_sample);
}

ead::NodeptrT<PybindT> reconstruct_hidden (RBM& rbm, ead::NodeptrT<PybindT> hidden)
{
	ead::NodeptrT<PybindT> visible_dist = rbm.prop_down(hidden);
	ead::NodeptrT<PybindT> visible_sample = eqns::one_binom(visible_dist);
	return rbm(visible_sample);
}

}

#endif // MODL_RBM_HPP
