#include "layr/dense.hpp"
#include "layr/rbm.hpp"

#ifndef RCN_DBN_TRAINER_HPP
#define RCN_DBN_TRAINER_HPP

namespace trainer
{

static bool is_dbn (layr::SequentialModel& model)
{
	auto layers = model.get_layers();

	size_t n = layers.size();
	size_t i = 0;
	// dbn can start with a series of dense/rbm layers with sigmoid activations
	for (; i < n - 2; ++i)
	{
		if (layers[i]->get_ltype() != layr::rbm_layer_key)
		{
			return false;
		}
	}
	// dbn must end with a dense or rbm layer with softmax activations
	auto rbm_ltype = layers[n - 2]->get_ltype();
	auto activation_ltype = layers[n - 1]->get_ltype();
	if ((rbm_ltype != layr::dense_layer_key &&
		dense_ltype != layr::rbm_layer_key) ||
		activation_ltype != layr::softmax_layer_key)
	{
		return false;
	}
	return true;
}

struct DBNTrainer final
{
	DBNTrainer (layr::SequentialModel& model,
		uint8_t batch_size,
		PybindT learning_rate = 1e-3,
		size_t n_cont_div = 10,
		eteq::VarptrT<PybindT> train_in = nullptr) :
		brain_(brain),
		caches_({})
	{
		if (nullptr == train_in)
		{
			train_in_ = eteq::VarptrT<PybindT>(eteq::Variable<PybindT>::get(
				0.0, teq::Shape({brain->get_ninput(), batch_size}), "train_in"));
		}
		else
		{
			train_in_ = train_in;
		}
		train_out_ = train_in_;
		auto layers = brain_->get_layers();
		for (layr::RBMptrT& rbm : layers)
		{
			RBMTrainer trainer(rbm, nullptr,
				batch_size, learning_rate,
				n_cont_div, train_out_);
			rbm_trainers_.push_back(trainer);
			train_out_ = rbm->prop_up(train_out_);
		}
	}

	std::vector<layr::Deltas> pretraining_functions (void) const
	{
		std::vector<layr::Deltas> pt_updates(rbm_trainers_.size());
		std::transform(rbm_trainers_.begin(), rbm_trainers_.end(),
			pt_updates.begin(),
			[](const RBMTrainer& trainer)
			{
				return trainer.updates_;
			});
		return pt_updates;
	}

	// todo: conform to a current trainer convention,
	// or make all trainers functions instead of class bundles
	std::pair<layr::Deltas,teq::TensptrT> build_finetune_functions (
		eteq::VarptrT<PybindT> train_out, PybindT learning_rate = 1e-3)
	{
		teq::TensptrT out_dist = (*brain_)(teq::TensptrT(train_in_));
		teq::TensptrT finetune_cost = egen::neg(
			egen::reduce_mean(egen::log(out_dist)));

		teq::TensptrT temp_diff = egen::sub(out_dist, teq::TensptrT(train_out));
		teq::TensptrT error = egen::reduce_mean(
			egen::pow(temp_diff,
				teq::TensptrT(eteq::Constant::get(2, temp_diff->shape()))));

		pbm::PathedMapT vmap = brain_->list_bases();
		layr::VariablesT vars;
		for (auto vpair : vmap)
		{
			if (eteq::VarptrT<PybindT> var = std::dynamic_pointer_cast<
				eteq::Variable<PybindT>>(vpair.first))
			{
				vars.push_back(var);
			}
		}
		layr::Deltas errs;
		layr::VarmapT connection;
		for (eteq::VarptrT<PybindT>& gp : vars)
		{
			auto next_gp = egen::sub(teq::TensptrT(gp), egen::mul(
				teq::TensptrT(eteq::Constant::get(learning_rate, gp->shape())),
				eteq::derive(finetune_cost, gp.get()))
			);
			errs.upkeep_.push_back(next_gp);
			connection.emplace(gp.get(), next_gp);
		}

		errs.actions_.push_back(
			[connection](eteq::CacheSpace<PybindT>* caches)
			{
				layr::assign_all(caches, connection);
			});

		return {errs, error};
	}

	eteq::VarptrT<PybindT> train_in_;

	teq::TensptrT train_out_;

	layr::DBNptrT brain_;

	std::vector<RBMTrainer> rbm_trainers_;

	eteq::CacheSpace<PybindT> caches_;
};

}

#endif // RCN_DBN_TRAINER_HPP
