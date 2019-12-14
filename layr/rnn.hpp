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

		placeholder_connect(calc_insign());
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
		teq::ShapeSignature inshape = input->shape_sign();
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
			init_state_, teq::ShapeSignature(slice_shape));
		eteq::LinksT<PybindT> states;
		for (teq::DimT i = 0, nseq = inshape.at(seq_dim); i < nseq; ++i)
		{
			auto inslice = tenncor::slice(input, i, 1, seq_dim);
			state = activation_->connect(
				cell_->connect(tenncor::concat(inslice, state, 0)));
			states.push_back(state);
		}
		return tenncor::concat(states, seq_dim);
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

		this->input_ = nullptr;
		this->placeholder_connect(calc_insign());
	}

	teq::ShapeSignature calc_insign (void) const
	{
		if (nullptr != params_)
		{
			teq::NElemT n = params_->shape().n_elems();
			if (1 > n)
			{
				logs::warnf("multiple sequence dimensions (%d) "
					"specified in rnn parameter", (int) n);
			}
			auto rawdims = (PybindT*) params_->data();
			teq::RankT seq_dim = rawdims[0];
			teq::ShapeSignature insign = cell_->get_input_sign();
			teq::ShapeSignature outsign = cell_->get_output_sign();
			std::vector<teq::DimT> slist(insign.begin(), insign.end());
			if (slist.at(seq_dim) > 0 && outsign.at(seq_dim) > 0)
			{
				slist[seq_dim] -= outsign.at(seq_dim);
			}
			else
			{
				slist[seq_dim] = 0;
			}
			return teq::ShapeSignature(slist);
		}
		// sequence dimension can be any
		return teq::ShapeSignature();
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
