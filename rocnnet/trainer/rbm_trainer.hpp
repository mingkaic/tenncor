#include "ead/grader.hpp"

#include "rocnnet/modl/rbm.hpp"

#ifndef MODL_RBM_TRAINER_HPP
#define MODL_RBM_TRAINER_HPP

struct RBMTrainer
{
	RBMTrainer (modl::RBMptrT brain,
		ead::Session<double>& sess,
		ead::VarptrT<double> persistent,
		uint8_t batch_size,
		double learning_rate = 1e-3,
		size_t n_cont_div = 1,
		ade::NodeptrT<double> train_in = nullptr) :
		batch_size_(batch_size),
		brain_(brain),
		weight_(brain->get_weight()),
		hbias_(brain->get_hbias()),
		vbias_(brain->get_vbias()),
		sess_(&sess)
	{
		if (nullptr == train_in)
		{
			train_in_ = ead::make_variable_scalar<double>(
				0.0, ade::Shape({brain->get_ninput(), batch_size}), "train_in"));
		}
		else
		{
			train_in_ = train_in;
		}
		// if persistent not available use Contrastive Divergence (CD)
		if (nullptr == persistent)
		{
			persistent_ = eqns::one_binom(brain->prop_up(train_in_));
		}
		// otherwise use Persistent CD
		// (initialize from the old state of the chain)
		else
		{
			persistent_ = persistent;
		}

		// chain length is n_cont_div
		ade::TensptrT chain_segment = eqns::one_binom(brain->reconstruct_hidden(persistent_));
		assert(n_cont_div > 0);
		for (size_t i = 0; i < n_cont_div - 1; ++i)
		{
			chain_segment = eqns::one_binom(brain->reconstruct_hidden(chain_segment));
		}

		// use operational optimization to recover presig and vis nodes
		ade::TensptrT weighed = age::matmul(
			chain_segment, age::transpose(weight_));
		ade::TensptrT presig_vis = eqns::weighed_bias_add(weighed, vbias_);
		ade::TensptrT final_visible_dist = eqns::sigmoid(presig_vis);
		ade::TensptrT chain_end = eqns::one_binom(final_visible_dist);

		cost_ = age::sub(age::reduce_mean(free_energy(train_in_)),
			age::reduce_mean(free_energy(chain_end)));

		ade::TensptrT dW = ead::derive(cost_, weight_.get());
		ade::TensptrT dhb = ead::derive(cost_, hbias_.get());
		ade::TensptrT dvb = ead::derive(cost_, vbias_.get());

		auto next_weight = age::sub(ade::TensptrT(weight_),
			age::mul(ade::TensptrT(
				ead::Constant::get(learning_rate, dW->shape())), dW));
		auto next_hbias = age::sub(ade::TensptrT(hbias_),
			age::mul(ade::TensptrT(
				ead::Constant::get(learning_rate, dhb->shape())), dhb));
		auto next_vbias = age::sub(ade::TensptrT(vbias_),
			age::mul(ade::TensptrT(
				ead::Constant::get(learning_rate, dvb->shape())), dvb));
		updates_.upkeep_ = {
			next_weight,
			next_hbias,
			next_vbias
		};
		eqns::VarmapT connection = {
			{weight_.get(), next_weight},
			{hbias_.get(), next_hbias},
			{vbias_.get(), next_vbias},
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
			auto next_persistent = eqns::one_binom(brain->reconstruct_hidden(chain_segment));
			updates_.upkeep_.push_back(next_persistent);
			connection.emplace(persistent.get(), next_persistent);

			monitoring_cost_ = get_pseudo_likelihood_cost(train_in_);
		}

		updates_.actions_.push_back(
			[connection](ead::Session<double>& sess)
			{
				eqns::assign_all(sess, connection);
			});
		for (ead::NodeptrT<double>& up : updates_.upkeep_)
		{
			sess_->track(up);
		}
	}

	// input a 2-D vector of shape <n_input, n_batch> return monitor cost
	double train (std::vector<double>& train_in)
	{
		if (auto var = dynamic_cast<ead::Variable<double>*>(train_in_.get()))
		{
			*var = ead::get_tensor(
				train_in.data(), var->shape());
			caches_.update({var});
			updates_.assign(&caches_);

			return *(ead::eval<double>(monitoring_cost_)->data()) / batch_size_;
		}
		logs::fatal("cannot train RBM with non-native input");
	}

	uint8_t batch_size_;
	ade::TensptrT train_in_;
	modl::RBMptrT brain_;
	ade::TensptrT cost_;

	ade::TensptrT monitoring_cost_;
	eqns::Deltas updates_;

	ead::Session<double> sess_;

private:
	ade::TensptrT get_pseudo_likelihood_cost (ade::TensptrT input)
	{
		const ade::Shape& shape = input->shape();
		std::vector<double> zeros(shape.n_elems(), 0);
		zeros[0] = 1;
		ade::TensptrT one_i(ead::Constant::get(zeros, shape));

		ade::TensptrT xi = age::round(input); // xi = [0|1, ...]
		ade::TensptrT xi_flip = age::sub(one_i, xi);

		ade::TensptrT fe_xi = free_energy(xi);
		ade::TensptrT fe_xi_flip = free_energy(xi_flip);

		return age::reduce_mean(age::mul(
			ade::TensptrT(ead::Constant::get(brain_->get_ninput(), fe_xi->shape())),
			age::log(eqns::sigmoid(age::sub(fe_xi_flip, fe_xi)))));
	}

	ade::TensptrT get_reconstruction_cost (ade::TensptrT input, ade::TensptrT visible_dist)
	{
		ade::TensptrT p_success = age::mul(input, age::log(visible_dist));
		ade::TensptrT p_not = age::mul(
			age::sub(ade::TensptrT(ead::Constant::get(1, input->shape())), input),
			age::log(age::sub(ade::TensptrT(
				ead::Constant::get(1, visible_dist->shape())), visible_dist)));
		return age::reduce_mean(
			age::reduce_sum(age::transpose(age::add(p_success, p_not)), 1));
	}

	ade::TensptrT free_energy (ade::TensptrT sample)
	{
		ade::TensptrT vbias_term = age::matmul(sample,
			age::transpose(ade::TensptrT(vbias_)));
		// <x, y> @ <z, x> + z -> <z, y>
		ade::TensptrT weighed = age::matmul(sample, ade::TensptrT(weight_));
		ade::TensptrT wx_b = eqns::weighed_bias_add(weighed, ade::TensptrT(hbias_));
		ade::TensptrT hidden_term = age::reduce_sum(
			age::log(age::add(
				ade::TensptrT(ead::Constant::get(1.0, wx_b->shape())),
				age::exp(wx_b)
			)), 0, 1);
		return age::neg(age::add(hidden_term, vbias_term));
	}

	ead::VarptrT<double> weight_;
	ead::VarptrT<double> hbias_;
	ead::VarptrT<double> vbias_;

	ade::TensptrT persistent_;
};

#endif // MODL_RBM_TRAINER_HPP
