///
/// gru.hpp
/// layr
///
/// Purpose:
/// Implement gated recurrent unit
///

#include "layr/dense.hpp"
#include "layr/ulayer.hpp"

#ifndef LAYR_GRU_HPP
#define LAYR_GRU_HPP

namespace layr
{

/// Update gate fully connected layer label
const std::string update_key = "update_gate";

/// Reset gate fully connected layer label
const std::string reset_key = "reset_gate";

/// Hidden gate fully connected layer label
const std::string hgate_key = "hidden_gate";

/// Builder implementation for gru layer
struct GRUBuilder final : public iLayerBuilder
{
	GRUBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override {} // gru has no tensor

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

/// Identifier for gated recurrent unit
const std::string gru_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "gru",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<GRUBuilder>(label);
});

struct GRU final : public iLayer
{
	GRU (teq::DimT nhidden, teq::DimT ninput,
		layr::InitF<PybindT> weight_init,
		layr::InitF<PybindT> bias_init,
		const std::string& label) :
		label_(label),
		ugate_(std::make_shared<Dense>(
			nhidden, teq::Shape({(teq::DimT) (nhidden + ninput)}),
			weight_init, bias_init, nullptr, update_key)),
		rgate_(std::make_shared<Dense>(
			nhidden, teq::Shape({(teq::DimT) (nhidden + ninput)}),
			weight_init, bias_init, nullptr, reset_key)),
		hgate_(std::make_shared<Dense>(
			nhidden, teq::Shape({(teq::DimT) (nhidden + ninput)}),
			weight_init, bias_init, nullptr, hgate_key))
	{
		tag_sublayers();
	}

	GRU (DenseptrT ugate, DenseptrT rgate,
		DenseptrT hgate, const std::string& label) :
		label_(label),
		ugate_(ugate),
		rgate_(rgate),
		hgate_(hgate)
	{
		tag_sublayers();
	}

	GRU (const GRU& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	GRU& operator = (const GRU& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	GRU (GRU&& other) = default;

	GRU& operator = (GRU&& other) = default;

	/// Return deep copy of this layer with prefixed label
	GRU* clone (std::string label_prefix = "") const
	{
		return static_cast<GRU*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	size_t get_ninput (void) const override
	{
		return ugate_->get_ninput() - ugate_->get_noutput();
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		return ugate_->get_noutput();
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return gru_layer_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		auto out = ugate_->get_contents();
		auto rcontents = rgate_->get_contents();
		auto hcontents = hgate_->get_contents();
		out.insert(out.end(), rcontents.begin(), rcontents.end());
		out.insert(out.end(), hcontents.begin(), hcontents.end());
		return out;
	}

	/// Implementation of iLayer
	NodeptrT connect (NodeptrT input) const override
	{
		// expecting input of shape <nunits, sequence length, ANY>
		// sequence is dimension 1
		teq::Shape inshape = input->shape();
		auto state = eteq::make_constant_scalar<PybindT>(0,
			teq::Shape({(teq::DimT) this->get_noutput()}));
		eteq::NodesT<PybindT> states;
		for (teq::DimT i = 0, nseq = inshape.at(1); i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, 1);
			state = cell_connect(inslice, state);
			states.push_back(state);
		}
		return tenncor::concat(states, 1);
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new GRU(*this, label_prefix);
	}

	void tag_sublayers (void)
	{
		auto ugate_subs = ugate_->get_contents();
		for (auto& sub : ugate_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(ugate_->get_ltype(), ugate_->get_label(), 0));
			}
		}
		auto rgate_subs = rgate_->get_contents();
		for (auto& sub : rgate_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(rgate_->get_ltype(), rgate_->get_label(), 1));
			}
		}
		auto hgate_subs = hgate_->get_contents();
		for (auto& sub : hgate_subs)
		{
			if (sub)
			{
				tag(sub, LayerId(hgate_->get_ltype(), hgate_->get_label(), 2));
			}
		}
	}

	void copy_helper (const GRU& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		ugate_ = DenseptrT(other.ugate_->clone(label_prefix));
		rgate_ = DenseptrT(other.rgate_->clone(label_prefix));
		hgate_ = DenseptrT(other.hgate_->clone(label_prefix));
		tag_sublayers();
	}

	NodeptrT cell_connect (NodeptrT x, NodeptrT state) const
	{
		NodeptrT xc = tenncor::concat(x, state, 0);
		auto update = tenncor::sigmoid(ugate_->connect(xc));
		auto reset = tenncor::sigmoid(rgate_->connect(xc));
		auto hidden = tenncor::tanh(hgate_->connect(
			tenncor::concat(x, reset * state, 0)));
		return update * state + ((PybindT) 1 - update) * hidden;
	}

	std::string label_;

	DenseptrT ugate_;

	DenseptrT rgate_;

	DenseptrT hgate_;
};

/// Smart pointer of GRU layer
using GRUptrT = std::shared_ptr<GRU>;

}

#endif // LAYR_GRU_HPP
