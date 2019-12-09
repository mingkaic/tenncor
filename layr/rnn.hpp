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

/// RNN's initial state label
const std::string init_state_key = "init_state";

/// RNN parameter label
const std::string rnn_param_key = "rnn_param";

/// Builder implementation for recurrent layer
struct RNNBuilder final : public iLayerBuilder
{
	RNNBuilder (std::string label) : label_(label) {}

	/// Implementation of iLayerBuilder
	void set_tensor (teq::TensptrT tens, std::string target) override
	{
		if (target == init_state_key)
		{
			init_state_ = eteq::to_link<PybindT>(tens);
			return;
		}
		else if (target == rnn_param_key)
		{
			params_ = eteq::to_link<PybindT>(tens);
			return;
		}
		logs::warnf("attempt to create rnn layer "
			"with unknown tensor `%s` of label `%s`",
			tens->to_string().c_str(), target.c_str());
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

	LinkptrT init_state_;

	LinkptrT params_ = nullptr;

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
	RNN (teq::DimT nunits, teq::DimT ninput, UnaryptrT activation,
		layr::InitF<PybindT> weight_init, layr::InitF<PybindT> bias_init,
		teq::RankT seq_dim, const std::string& label) :
		label_(label),
		cell_(std::make_shared<Dense>(nunits,
			teq::Shape({(teq::DimT) (nunits + ninput)}),
			weight_init, bias_init, nullptr, "cell")),
		init_state_(eteq::to_link<PybindT>(
			eteq::make_variable<PybindT>(
				teq::Shape({nunits}), "init_state"))),
		activation_(activation),
		params_(eteq::make_constant_scalar<PybindT>(seq_dim, teq::Shape()))
	{
		tag_sublayers();
	}

	RNN (DenseptrT cell, UnaryptrT activation, LinkptrT init_state,
		LinkptrT params, const std::string& label) :
		label_(label), cell_(cell), init_state_(init_state),
		activation_(activation), params_(params)
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
		return cell_->get_ninput() - cell_->get_noutput();
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
		if (nullptr == params_)
		{
			out.push_back(nullptr);
		}
		else
		{
			out.push_back(params_->get_tensor());
		}
		return out;
	}

	/// Implementation of iLayer
	LinkptrT connect (LinkptrT input) const override
	{
		teq::Shape inshape = input->shape();
		teq::RankT seq_dim;
		if (nullptr != params_)
		{
			teq::NElemT n = params_->shape().n_elems();
			if (1 > n)
			{
				logs::warnf("multiple sequence dimensions (%d) "
					"specified in rnn parameter", (int) n);
			}
			auto rawdims = (PybindT*) params_->data();
			seq_dim = rawdims[0];
		}
		else
		{
			// take last non-single dimension as sequence dimension
			auto slist = teq::narrow_shape(inshape);
			seq_dim = slist.empty() ? 1 : slist.size() - 1;
		}
		if (seq_dim == 0)
		{
			logs::warn("spliting input across 0th dimension... "
				"dense connection might not match");
		}
		std::vector<teq::DimT> slice_shape(inshape.begin(), inshape.end());
		slice_shape[seq_dim] = 1;
		LinkptrT state = tenncor::best_extend(
			init_state_, teq::Shape(slice_shape));
		eteq::LinksT<PybindT> states;
		for (teq::DimT i = 0, nseq = inshape.at(seq_dim); i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, seq_dim);
			state = activation_->connect(
				cell_->connect(tenncor::concat(inslice, state, 0)));
			states.push_back(state);
		}
		auto output = tenncor::concat(states, seq_dim);
		recursive_tag(output->get_tensor(), {
			input->get_tensor().get(),
		}, LayerId());
		return output;
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
		if (nullptr != params_)
		{
			tag(params_->get_tensor(), LayerId(rnn_param_key));
		}
	}

	void copy_helper (const RNN& other, std::string label_prefix = "")
	{
		label_ = label_prefix + other.label_;
		cell_ = DenseptrT(other.cell_->clone(label_prefix));
		init_state_ = LinkptrT(other.init_state_->clone());
		activation_ = UnaryptrT(other.activation_->clone(label_prefix));
		if (nullptr != other.params_)
		{
			params_ = LinkptrT(other.params_->clone());
		}
		tag_sublayers();
	}

	std::string label_;

	DenseptrT cell_;

	LinkptrT init_state_;

	UnaryptrT activation_;

	LinkptrT params_;
};

/// Smart pointer of recurrent model
using RNNptrT = std::shared_ptr<RNN>;

}

#endif // LAYR_RNN_HPP
