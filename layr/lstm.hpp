///
/// lstm.hpp
/// layr
///
/// Purpose:
/// Implement long-short term memory
///

#include "layr/dense.hpp"
#include "layr/ulayer.hpp"

#ifndef LAYR_LSTM_HPP
#define LAYR_LSTM_HPP

namespace layr
{

/// Gate fully connected layer label
const std::string gate_key = "gate";

/// Forget fully connected layer label
const std::string forget_key = "forget";

/// Input Gate fully connected layer label
const std::string ingate_key = "ingate";

/// Output Gate fully connected layer label
const std::string outgate_key = "outgate";

/// Builder implementation for lstm layer
struct LSTMBuilder final : public iLayerBuilder
{
	LSTMBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override {} // lstm has no tensor

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override
	{
		layers_.push_back(layer);
	}

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	std::vector<LayerptrT> layers_;

	std::string label_;
};

/// Identifier for long-short term memory
const std::string lstm_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "lstm",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<LSTMBuilder>(label);
});

struct LSTM final : public iLayer
{
	LSTM (teq::DimT nhidden, teq::DimT ninput,
		layr::InitF<PybindT> weight_init,
		layr::InitF<PybindT> bias_init,
		const std::string& label) :
		label_(label),
		gate_(std::make_shared<Dense>(
			nhidden, teq::Shape({(teq::DimT) (nhidden + ninput)}),
			weight_init, bias_init, nullptr, gate_key)),
		forget_(std::make_shared<Dense>(
			nhidden, teq::Shape({(teq::DimT) (nhidden + ninput)}),
			weight_init, bias_init, nullptr, forget_key)),
		ingate_(std::make_shared<Dense>(
			nhidden, teq::Shape({(teq::DimT) (nhidden + ninput)}),
			weight_init, bias_init, nullptr, ingate_key)),
		outgate_(std::make_shared<Dense>(
			nhidden, teq::Shape({(teq::DimT) (nhidden + ninput)}),
			weight_init, bias_init, nullptr, outgate_key))
	{
		tag_sublayers();
	}

	LSTM (DenseptrT gate, DenseptrT forget,
		DenseptrT ingate, DenseptrT outgate,
		const std::string& label) :
		label_(label),
		gate_(gate),
		forget_(forget),
		ingate_(ingate),
		outgate_(outgate)
	{
		tag_sublayers();
	}

	LSTM (const LSTM& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	LSTM& operator = (const LSTM& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	LSTM (LSTM&& other) = default;

	LSTM& operator = (LSTM&& other) = default;

	/// Return deep copy of this layer with prefixed label
	LSTM* clone (std::string label_prefix = "") const
	{
		return static_cast<LSTM*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	size_t get_ninput (void) const override
	{
		return gate_->get_ninput() - gate_->get_noutput();
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		return gate_->get_noutput();
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return lstm_layer_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		auto out = gate_->get_contents();
		auto fcontents = forget_->get_contents();
		auto icontents = ingate_->get_contents();
		auto ocontents = outgate_->get_contents();
		out.insert(out.end(), fcontents.begin(), fcontents.end());
		out.insert(out.end(), icontents.begin(), icontents.end());
		out.insert(out.end(), ocontents.begin(), ocontents.end());
		return out;
	}

	/// Implementation of iLayer
	LinkptrT connect (LinkptrT input) const override
	{
		// expecting input of shape <nunits, sequence length, ANY>
		// sequence is dimension 1
		teq::Shape inshape = input->shape();
		teq::Shape stateshape({(teq::DimT) this->get_noutput()});
		auto prevstate = eteq::make_constant_scalar<PybindT>(0, stateshape);
		auto prevhidden = eteq::make_constant_scalar<PybindT>(0, stateshape);
		eteq::LinksT<PybindT> states;
		for (teq::DimT i = 0, nseq = inshape.at(1); i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, 1);
			auto nexts = cell_connect(inslice, prevstate, prevhidden);
			prevstate = nexts.first;
			prevhidden = nexts.second;
			states.push_back(prevhidden);
		}
		auto output = tenncor::concat(states, 1);
		recursive_tag(output->get_tensor(), {
			input->get_tensor().get(),
		}, LayerId());
		return output;
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new LSTM(*this, label_prefix);
	}

	void tag_sublayers (void)
	{
		auto gate_subs = gate_->get_contents();
		for (auto& sub : gate_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(gate_->get_ltype(), gate_->get_label(), 0));
			}
		}
		auto forget_subs = forget_->get_contents();
		for (auto& sub : forget_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(forget_->get_ltype(), forget_->get_label(), 1));
			}
		}
		auto ingate_subs = ingate_->get_contents();
		for (auto& sub : ingate_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(ingate_->get_ltype(), ingate_->get_label(), 2));
			}
		}
		auto outgate_subs = outgate_->get_contents();
		for (auto& sub : outgate_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(outgate_->get_ltype(), outgate_->get_label(), 3));
			}
		}
	}

	void copy_helper (const LSTM& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		gate_ = DenseptrT(other.gate_->clone(label_prefix));
		forget_ = DenseptrT(other.forget_->clone(label_prefix));
		ingate_ = DenseptrT(other.ingate_->clone(label_prefix));
		outgate_ = DenseptrT(other.outgate_->clone(label_prefix));
		tag_sublayers();
	}

	std::pair<LinkptrT,LinkptrT> cell_connect (LinkptrT x,
		LinkptrT prev_state, LinkptrT prev_hidden) const
	{
		LinkptrT xc = tenncor::concat(x, prev_hidden, 0);

		auto gate = tenncor::tanh(gate_->connect(xc));
		auto input = tenncor::sigmoid(ingate_->connect(xc));
		auto forget = tenncor::sigmoid(forget_->connect(xc));
		auto output = tenncor::sigmoid(outgate_->connect(xc));
		auto state = gate * input + prev_state * forget;
		return {state, state * output};
	}

	std::string label_;

	DenseptrT gate_;

	DenseptrT forget_;

	DenseptrT ingate_;

	DenseptrT outgate_;
};

/// Smart pointer of LSTM layer
using LSTMptrT = std::shared_ptr<LSTM>;

}

#endif // LAYR_LSTM_HPP
