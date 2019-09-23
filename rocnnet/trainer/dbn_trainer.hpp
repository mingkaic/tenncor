#include "modl/dbn.hpp"

#include "rocnnet/trainer/rbm_trainer.hpp"

struct DBNTrainer final
{
	DBNTrainer (modl::DBNptrT brain,
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
		for (modl::RBMptrT& rbm : layers)
		{
			RBMTrainer trainer(rbm, nullptr,
				batch_size, learning_rate,
				n_cont_div, train_out_);
			rbm_trainers_.push_back(trainer);
			train_out_ = rbm->prop_up(train_out_);
		}
	}

	std::vector<eqns::Deltas> pretraining_functions (void) const
	{
		std::vector<eqns::Deltas> pt_updates(rbm_trainers_.size());
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
	std::pair<eqns::Deltas,teq::TensptrT> build_finetune_functions (
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
		eqns::VariablesT vars;
		for (auto vpair : vmap)
		{
			if (eteq::VarptrT<PybindT> var = std::dynamic_pointer_cast<
				eteq::Variable<PybindT>>(vpair.first))
			{
				vars.push_back(var);
			}
		}
		eqns::Deltas errs;
		eqns::VarmapT connection;
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
				eqns::assign_all(caches, connection);
			});

		return {errs, error};
	}

	eteq::VarptrT<PybindT> train_in_;

	teq::TensptrT train_out_;

	modl::DBNptrT brain_;

	std::vector<RBMTrainer> rbm_trainers_;

	eteq::CacheSpace<PybindT> caches_;
};
