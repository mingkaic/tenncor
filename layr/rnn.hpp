///
/// rnn.hpp
/// layr
///
/// Purpose:
/// Implement vanilla recurrent neural net layer
///

#include "layr/dense.hpp"
#include "layr/ulayer.hpp"

#ifndef LAYR_RNN_HPP
#define LAYR_RNN_HPP

namespace layr
{

/// Recurrent layer's initial state label
const std::string init_state_key = "init_state";

/// Builder implementation for recurrent layer
struct RNNBuilder final : public iLayerBuilder
{
	RNNBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override
	{
		if (target == init_state_key)
		{
			init_state_ = eteq::to_node<PybindT>(tens);
			return;
		}
	}

	/// Implementation of iLayerBuilder
	void set_sublayer (LayerptrT layer) override
	{
		layers_.push_back(layer);
	}

	/// Implementation of iLayerBuilder
	LayerptrT build (void) const override;

private:
	std::string label_;

	NodeptrT init_state_;

	std::vector<LayerptrT> layers_;
};

/// Identifier for recurrent layer
const std::string rnn_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "rnn", // todo: rename
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<RNNBuilder>(label);
});

/// Layer implementation that applies recurrent cells (cells applied at each step of input)
struct RNN final : public iLayer
{
	RNN (teq::DimT nunits,
		UnaryptrT activation,
		layr::InitF<PybindT> weight_init,
		layr::InitF<PybindT> bias_init,
		const std::string& label) :
		label_(label),
		cell_(std::make_shared<Dense>(nunits, teq::Shape({nunits}),
			weight_init, bias_init, nullptr, "cell")),
		init_state_(eteq::convert_to_node(
			eteq::make_variable<PybindT>(
				teq::Shape({nunits}), "init_state"))),
		activation_(activation)
	{
		tag_sublayers();
	}

	RNN (DenseptrT cell, UnaryptrT activation, NodeptrT init_state,
		const std::string& label) :
		label_(label),
		cell_(cell),
		init_state_(init_state),
		activation_(activation)
	{
		tag_sublayers();
	}

	RNN (const RNN& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	RNN& operator = (const RNN& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	RNN (RNN&& other) = default;

	RNN& operator = (RNN&& other) = default;

	/// Return deep copy of this model with prefixed label
	RNN* clone (std::string label_prefix = "") const
	{
		return static_cast<RNN*>(this->clone_impl(label_prefix));
	}

	/// Implementation of iLayer
	size_t get_ninput (void) const override
	{
		return cell_->get_ninput();
	}

	/// Implementation of iLayer
	size_t get_noutput (void) const override
	{
		return cell_->get_noutput();
	}

	/// Implementation of iLayer
	std::string get_ltype (void) const override
	{
		return rnn_layer_key;
	}

	/// Implementation of iLayer
	std::string get_label (void) const override
	{
		return label_;
	}

	/// Implementation of iLayer
	teq::TensptrsT get_contents (void) const override
	{
		teq::TensptrsT out = cell_->get_contents();
		auto act_contents = activation_->get_contents();
		out.insert(out.end(), act_contents.begin(), act_contents.end());
		out.push_back(init_state_->get_tensor());
		return out;
	}

	/// Implementation of iLayer
	NodeptrT connect (NodeptrT input) const override
	{
		// expecting input of shape <nunits, sequence length, ANY>
		// sequence is dimension 1
		teq::Shape inshape = input->shape();
		NodeptrT prevstate = tenncor::best_extend(
			init_state_, teq::Shape({
				(teq::DimT) get_ninput(), 1, inshape.at(2),
			}));
		eteq::NodesT<PybindT> states;
		for (teq::DimT i = 0, nseq = inshape.at(1); i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, 1);
			prevstate = activation_->connect(
				inslice + cell_->connect(prevstate));
			states.push_back(prevstate);
		}
		return tenncor::concat(states, 1);
	}

private:
	iLayer* clone_impl (const std::string& label_prefix) const override
	{
		return new RNN(*this, label_prefix);
	}

	void tag_sublayers (void)
	{
		tag(init_state_->get_tensor(), LayerId(init_state_key));
		auto subs = cell_->get_contents();
		for (auto& sub : subs)
		{
			if (sub)
			{
				tag(sub, LayerId(cell_->get_ltype(), cell_->get_label(), 0));
			}
		}

		auto activation_subs = activation_->get_contents();
		for (auto& sub : activation_subs)
		{
			tag(sub, LayerId(activation_->get_ltype(),
				activation_->get_label(), 2));
		}
	}

	void copy_helper (const RNN& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		cell_ = DenseptrT(other.cell_->clone(label_prefix));
		init_state_ = NodeptrT(other.init_state_->clone());
		activation_ = UnaryptrT(other.activation_->clone(label_prefix));
		tag_sublayers();
	}

	std::string label_;

	DenseptrT cell_;

	NodeptrT init_state_;

	UnaryptrT activation_;
};

/// Smart pointer of recurrent model
using RNNptrT = std::shared_ptr<RNN>;

}

#endif // LAYR_RNN_HPP
