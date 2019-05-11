#include <functional>
#include <memory>

#include "prx/api.hpp"

#include "rocnnet/eqns/helper.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifndef MODL_RBM_HPP
#define MODL_RBM_HPP

namespace modl
{

struct RBM final : public iMarshalSet
{
	RBM (uint8_t n_input, uint8_t n_hidden, std::string label) :
		iMarshalSet(label)
	{
		ade::Shape shape({n_hidden, n_input});
		size_t nw = shape.n_elems();

		PybindT bound = 4 * std::sqrt(6.0 / (n_hidden + n_input));
		std::uniform_real_distribution<PybindT> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(ead::get_engine());
		};
		std::vector<PybindT> wdata(nw);
		std::generate(wdata.begin(), wdata.end(), gen);

		ead::VarptrT<PybindT> weight = ead::make_variable<PybindT>(
			wdata.data(), shape, "weight");
		ead::VarptrT<PybindT> hbias = ead::make_variable_scalar<PybindT>(
			0.0, ade::Shape({n_hidden}), "hbias");
		ead::VarptrT<PybindT> vbias = ead::make_variable_scalar<PybindT>(
			0.0, ade::Shape({n_input}), "vbias");
		weight_ = std::make_shared<MarshalVar>(weight);
		hbias_ = std::make_shared<MarshalVar>(hbias);
		vbias_ = std::make_shared<MarshalVar>(vbias);
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
	ead::NodeptrT<PybindT> prop_up (ead::NodeptrT<PybindT> input)
	{
		// prop forward
		// weight is <n_hidden, n_input>
		// in is <n_input, ?>
		// out = in @ weight, so out is <n_hidden, ?>
		return age::sigmoid(prx::fully_connect({input},
			{ead::convert_to_node(weight_->var_)},
			ead::convert_to_node(hbias_->var_)));
	}

	// input of shape <n_hidden, n_batch>
	ead::NodeptrT<PybindT> prop_down (ead::NodeptrT<PybindT> hidden)
	{
		// weight is <n_hidden, n_input>
		// in is <n_hidden, ?>
		// out = in @ weight.T, so out is <n_input, ?>
		return age::sigmoid(prx::fully_connect({hidden},
			{age::transpose(ead::convert_to_node(weight_->var_))},
			ead::convert_to_node(vbias_->var_)));
	}

	// recreate input using hidden distribution
	// output shape of input->shape()
	ead::NodeptrT<PybindT> reconstruct_visible (ead::NodeptrT<PybindT> input)
	{
		ead::NodeptrT<PybindT> hidden_dist = prop_up(input);
		ead::NodeptrT<PybindT> hidden_sample = eqns::one_binom(hidden_dist);
		return prop_down(hidden_sample);
	}

	ead::NodeptrT<PybindT> reconstruct_hidden (ead::NodeptrT<PybindT> hidden)
	{
		ead::NodeptrT<PybindT> visible_dist = prop_down(hidden);
		ead::NodeptrT<PybindT> visible_sample = eqns::one_binom(visible_dist);
		return prop_up(visible_sample);
	}

	uint8_t get_ninput (void) const
	{
		return weight_->var_->shape().at(1);
	}

	uint8_t get_noutput (void) const
	{
		return weight_->var_->shape().at(0);
	}

	MarsarrT get_subs (void) const override
	{
		return {weight_, hbias_, vbias_};
	}

	ead::VarptrT<PybindT> get_weight (void) const
	{
		return weight_->var_;
	}

	ead::VarptrT<PybindT> get_hbias (void) const
	{
		return hbias_->var_;
	}

	ead::VarptrT<PybindT> get_vbias (void) const
	{
		return vbias_->var_;
	}

private:
	void copy_helper (const RBM& other)
	{
		weight_ = std::make_shared<MarshalVar>(*other.weight_);
		hbias_ = std::make_shared<MarshalVar>(*other.hbias_);
		vbias_ = std::make_shared<MarshalVar>(*other.vbias_);
	}

	iMarshaler* clone_impl (void) const override
	{
		return new RBM(*this);
	}

	MarVarsptrT weight_;

	MarVarsptrT hbias_;

	MarVarsptrT vbias_;
};

using RBMptrT = std::shared_ptr<RBM>;

}

#endif // MODL_RBM_HPP
