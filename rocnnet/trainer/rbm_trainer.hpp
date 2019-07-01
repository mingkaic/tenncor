#include "rocnnet/modl/rbm.hpp"

#ifndef MODL_RBM_TRAINER_HPP
#define MODL_RBM_TRAINER_HPP

namespace trainer
{

// recreate input using hidden distribution
// output shape of input->shape()
ead::NodeptrT<PybindT> reconstruct_visible (modl::RBM& rbm,
	ead::NodeptrT<PybindT> input, modl::NonLinearsT nonlins)
{
	ead::NodeptrT<PybindT> hidden_dist = rbm(input, nonlins);
	ead::NodeptrT<PybindT> hidden_sample = tenncor::random::rand_binom_one(hidden_dist);
	return rbm.prop_down(hidden_sample, nonlins);
}

ead::NodeptrT<PybindT> reconstruct_hidden (modl::RBM& rbm,
	ead::NodeptrT<PybindT> hidden, modl::NonLinearsT nonlins)
{
	ead::NodeptrT<PybindT> visible_dist = rbm.prop_down(hidden, nonlins);
	ead::NodeptrT<PybindT> visible_sample = tenncor::random::rand_binom_one(visible_dist);
	return rbm(visible_sample, nonlins);
}

struct RBMTrainer
{
	RBMTrainer (modl::RBMptrT brain,
		modl::NonLinearsT nonlinearities,
		ead::iSession& sess,
		ead::VarptrT<PybindT> persistent,
		uint8_t batch_size,
		PybindT learning_rate = 1e-3,
		size_t n_cont_div = 1,
		ead::NodeptrT<PybindT> train_in = nullptr) :
		brain_(brain),
		sess_(&sess)
	{
		if (brain->layers_.size() != 1)
		{
			logs::error("rbm training only operates on the first layer");
		}
		ead::VarptrT<PybindT> weight = brain_->layers_[0].weight_->var_;
		ead::VarptrT<PybindT> hbias = brain_->layers_[0].hbias_->var_;
		ead::VarptrT<PybindT> vbias = brain_->layers_[0].vbias_->var_;

		if (nullptr == train_in)
		{
			train_in_ = ead::convert_to_node(
				ead::make_variable_scalar<PybindT>(0.0, ade::Shape({
					brain->get_ninput(),
					batch_size,
				}), "train_in"));
		}
		else
		{
			train_in_ = train_in;
		}
		// if persistent not available use Contrastive Divergence (CD)
		if (nullptr == persistent)
		{
			persistent_ = tenncor::random::rand_binom_one(
				(*brain)(train_in_, nonlinearities));
		}
		// otherwise use Persistent CD
		// (initialize from the old state of the chain)
		else
		{
			persistent_ = persistent;
		}

		// chain length is n_cont_div
		auto chain_segment = tenncor::random::rand_binom_one(
			reconstruct_hidden(*brain, persistent_, nonlinearities));
		assert(n_cont_div > 0);
		for (size_t i = 0; i < n_cont_div - 1; ++i)
		{
			chain_segment = tenncor::random::rand_binom_one(
				reconstruct_hidden(*brain, chain_segment, nonlinearities));
		}

		// use operational optimization to recover presig and vis nodes
		auto presig_vis = tenncor::nn::fully_connect(
			{chain_segment},
			{tenncor::transpose(ead::convert_to_node(weight))},
			ead::convert_to_node(vbias));
		auto final_visible_dist = tenncor::sigmoid(presig_vis);
		auto chain_end = tenncor::random::rand_binom_one(final_visible_dist);

		cost_ = tenncor::sub(tenncor::reduce_mean(free_energy(train_in_)),
			tenncor::reduce_mean(free_energy(chain_end)));

		auto dW = ead::derive(cost_, ead::convert_to_node(weight));
		auto dhb = ead::derive(cost_, ead::convert_to_node(hbias));
		auto dvb = ead::derive(cost_, ead::convert_to_node(vbias));

		auto next_weight = tenncor::sub(ead::convert_to_node(weight),
			tenncor::mul(ead::make_constant_scalar(
				learning_rate, dW->shape()), dW));
		auto next_hbias = tenncor::sub(ead::convert_to_node(hbias),
			tenncor::mul(ead::make_constant_scalar(
				learning_rate, dhb->shape()), dhb));
		auto next_vbias = tenncor::sub(ead::convert_to_node(vbias),
			tenncor::mul(ead::make_constant_scalar(
				learning_rate, dvb->shape()), dvb));
		eqns::AssignsT assigns = {
			eqns::VarAssign{"", weight, next_weight},
			eqns::VarAssign{"", hbias, next_hbias},
			eqns::VarAssign{"", vbias, next_vbias},
		};

		if (nullptr == persistent)
		{
			// reconstruction cost
			monitoring_cost_ = get_reconstruction_cost(
				train_in_, final_visible_dist);
		}
		else
		{
			// pseudo-likelihood
			auto next_persistent = tenncor::random::rand_binom_one(
				reconstruct_hidden(*brain, chain_segment, nonlinearities));
			assigns.push_back(
				eqns::VarAssign{"", persistent, next_persistent});

			monitoring_cost_ = get_pseudo_likelihood_cost(train_in_);
		}
		updates_.push_back(assigns);

		ade::TensT track_batch;
		track_batch.reserve(assigns.size());
		std::transform(assigns.begin(), assigns.end(),
			std::back_inserter(track_batch),
			[](eqns::VarAssign& assign)
			{
				return assign.source_->get_tensor();
			});
		sess_->track(track_batch);
	}

	// input a 2-D vector of shape <n_input, n_batch> return monitor cost
	PybindT train (std::vector<PybindT>& train_in)
	{
		auto var = dynamic_cast<ead::VariableNode<PybindT>*>(train_in_.get());
		if (nullptr == var)
		{
			logs::fatal("cannot train RBM with non-native input");
		}
		ade::Shape train_shape = var->shape();
		var->assign(train_in.data(), train_shape);
		assign_groups(updates_,
			[this](std::unordered_set<ade::iTensor*>& updated)
			{
				this->sess_->update(updated);
			});

		return monitoring_cost_->data()[0] / train_shape.at(1);
	}

	ead::NodeptrT<PybindT> train_in_;
	modl::RBMptrT brain_;
	ead::NodeptrT<PybindT> cost_;
	ead::NodeptrT<PybindT> monitoring_cost_;

	eqns::AssignGroupsT updates_;
	ead::iSession* sess_;

private:
	ead::NodeptrT<PybindT> get_pseudo_likelihood_cost (ead::NodeptrT<PybindT> input)
	{
		const ade::Shape& shape = input->shape();
		std::vector<PybindT> zeros(shape.n_elems(), 0);
		zeros[0] = 1;
		auto one_i = ead::make_constant(zeros.data(), shape);

		ead::NodeptrT<PybindT> xi = tenncor::round(input); // xi = [0|1, ...]
		ead::NodeptrT<PybindT> xi_flip = tenncor::sub(one_i, xi);

		ead::NodeptrT<PybindT> fe_xi = free_energy(xi);
		ead::NodeptrT<PybindT> fe_xi_flip = free_energy(xi_flip);

		return tenncor::reduce_mean(tenncor::mul(
			ead::make_constant_scalar<PybindT>(brain_->get_ninput(), fe_xi->shape()),
			tenncor::log(tenncor::sigmoid(tenncor::sub(fe_xi_flip, fe_xi)))));
	}

	ead::NodeptrT<PybindT> get_reconstruction_cost (
		ead::NodeptrT<PybindT> input, ead::NodeptrT<PybindT> visible_dist)
	{
		ead::NodeptrT<PybindT> p_success = tenncor::mul(
			input, tenncor::log(visible_dist));
		ead::NodeptrT<PybindT> p_not = tenncor::mul(
			tenncor::sub(ead::make_constant_scalar<PybindT>(
				1, input->shape()), input),
			tenncor::log(tenncor::sub(ead::make_constant_scalar<PybindT>(
				1, visible_dist->shape()), visible_dist)));
		return tenncor::reduce_mean(tenncor::reduce_sum_1d(
			tenncor::transpose(tenncor::add(p_success, p_not)), 1));
	}

	ead::NodeptrT<PybindT> free_energy (ead::NodeptrT<PybindT> sample)
	{
		ead::VarptrT<PybindT> weight = brain_->layers_[0].weight_->var_;
		ead::VarptrT<PybindT> hbias = brain_->layers_[0].hbias_->var_;
		ead::VarptrT<PybindT> vbias = brain_->layers_[0].vbias_->var_;

		auto vbias_term = tenncor::matmul(sample,
			tenncor::transpose(ead::convert_to_node(vbias)));
		// <x, y> @ <z, x> + z -> <z, y>
		auto wx_b = tenncor::nn::fully_connect(
			{sample},
			{ead::convert_to_node(weight)},
			ead::convert_to_node(hbias));
		auto hidden_term = tenncor::reduce_sum(
			tenncor::log(tenncor::add(
				ead::make_constant_scalar<PybindT>(1, wx_b->shape()),
				tenncor::exp(wx_b)
			)), 0, 1);
		return tenncor::neg(tenncor::add(vbias_term, hidden_term));
	}

	ead::NodeptrT<PybindT> persistent_;
};

}

#endif // MODL_RBM_TRAINER_HPP
