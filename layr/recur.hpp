///
/// recur.hpp
/// layr
///
/// Purpose:
/// Implement simple recurrent layer
///

#include "layr/dense.hpp"
#include "layr/ulayer.hpp"

#ifndef LAYR_RECUR_HPP
#define LAYR_RECUR_HPP

namespace layr
{

/// Recurrent layer's initial state label
const std::string init_state_key = "init_state";

/// Builder implementation for recurrent layer
struct RecurBuilder final : public iLayerBuilder
{
	RecurBuilder (std::string label) : label_(label) {}

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
const std::string rec_layer_key =
get_layer_reg().register_tagr(layers_key_prefix + "recur",
[](std::string label) -> LBuilderptrT
{
	return std::make_shared<RecurBuilder>(label);
});

/// Layer implementation that applies recurrent cells (cells applied at each step of input)
struct Recur final : public iLayer
{
	Recur (teq::DimT nunits,
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

	Recur (DenseptrT cell, UnaryptrT activation, NodeptrT init_state,
		const std::string& label) :
		label_(label),
		cell_(cell),
		init_state_(init_state),
		activation_(activation)
	{
		tag_sublayers();
	}

	Recur (const Recur& other,
		std::string label_prefix = "")
	{
		copy_helper(other, label_prefix);
	}

	Recur& operator = (const Recur& other)
	{
		if (this != &other)
		{
			copy_helper(other);
		}
		return *this;
	}

	Recur (Recur&& other) = default;

	Recur& operator = (Recur&& other) = default;

	/// Return deep copy of this model with prefixed label
	Recur* clone (std::string label_prefix = "") const
	{
		return static_cast<Recur*>(this->clone_impl(label_prefix));
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
		return rec_layer_key;
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
		return new Recur(*this, label_prefix);
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

	void copy_helper (const Recur& other, std::string label_prefix = "")
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
using RecurptrT = std::shared_ptr<Recur>;

}

#endif // LAYR_RECUR_HPP
