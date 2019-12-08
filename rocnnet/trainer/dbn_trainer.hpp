#include "layr/dense.hpp"
#include "layr/rbm.hpp"

#include "rocnnet/trainer/rbm_trainer.hpp"

#ifndef RCN_DBN_TRAINER_HPP
#define RCN_DBN_TRAINER_HPP

namespace trainer
{

// todo: only a portion of the sequential model needs to conform to these parameter (cover this case)
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
		rbm_ltype != layr::rbm_layer_key) ||
		activation_ltype != layr::softmax_layer_key)
	{
		return false;
	}
	return true;
}

// source: https://gist.github.com/yusugomori/4509019
// todo: add optimization options
struct DBNTrainer final
{
	DBNTrainer (layr::SequentialModel& model,
		uint8_t batch_size,
		PybindT pretrain_lr = 0.1,
		PybindT train_lr = 0.1,
		size_t cdk = 10,
		PybindT l2_reg = 0.,
		PybindT lr_scaling = 0.95) :
		input_size_(model.get_ninput()),
		output_size_(model.get_noutput()),
		batch_size_(batch_size)
	{
		if (false == is_dbn(model))
		{
			logs::fatal("cannot train non-dbn");
		}
		auto layers = model.get_layers();

		trainx_ = eteq::make_variable_scalar<PybindT>(
			0, teq::Shape({(teq::DimT) input_size_, batch_size}),
			"trainx");
		trainy_ = eteq::make_variable_scalar<PybindT>(
			0, teq::Shape({(teq::DimT) output_size_, batch_size}),
			"trainy");
		nlayers_ = layers.size() - 2;

		std::vector<layr::RBMptrT> rbm_layers;
		rbm_layers.reserve(nlayers_);
		for (size_t i = 0; i < nlayers_; ++i)
		{
			rbm_layers.push_back(
				std::static_pointer_cast<layr::RBM>(layers[i]));
		}
		auto dense_layer = layers[nlayers_];
		auto softmax_layer = layers.back();

		// setups:
		// general rbm sampling
		sample_pipes_.push_back(eteq::to_node<PybindT>(trainx_));
		for (size_t i = 0; i < nlayers_; ++i)
		{
			sample_pipes_.push_back(sample_v2h(
				*rbm_layers[i], sample_pipes_[i]));
		}

		// layer-wise rbm reconstruction
		teq::TensptrsT to_track;
		for (size_t i = 0; i < nlayers_; ++i)
		{
			auto& rbm = rbm_layers[i];
			auto& rx = sample_pipes_[i];
			auto& ry = sample_pipes_[i + 1];
			auto contents = rbm->get_contents();
			teq::TensSetT to_learn = {
				contents[0].get(),
				contents[1].get(),
				contents[4].get(),
			};
			CDChainIO io(rx, ry);
			layr::VarErrsT varerrs = cd_grad_approx(io, *rbm, cdk); // todo: add persistent option
			layr::AssignsT assigns;
			for (auto varerr : varerrs)
			{
				// if var is a weight or bias add assign with learning rate
				// otherwise assign directly
				auto next_var = estd::has(to_learn, varerr.first.get()) ?
					eteq::to_node<PybindT>(varerr.first) +
					pretrain_lr * varerr.second : varerr.second;
				assigns.push_back(layr::VarAssign{
					fmts::sprintf("rbm_d%s_%d",
						varerr.first->to_string().c_str(), i),
					varerr.first, next_var
				});
				to_track.push_back(next_var->get_tensor());
			}
			rupdates_.push_back(assigns);

			auto vhv = rbm->backward_connect(rbm->connect(rx));
			auto rcost = -tenncor::reduce_mean(
				tenncor::reduce_sum_1d(rx * tenncor::log(vhv) +
					((PybindT) 1 - rx) * tenncor::log((PybindT) 1 - vhv), 0));
			rcosts_.push_back(rcost);
			to_track.push_back(rcost->get_tensor());
		}
		to_track.push_back(sample_pipes_.back()->get_tensor());
		pretrain_sess_.track(to_track);

		// logistic layer training
		// todo: improve this adhoc way of training log layer
		auto contents = dense_layer->get_contents();
		auto& w = contents[0];
		auto& b = contents[1];
		auto final_out = softmax_layer->connect(dense_layer->connect(sample_pipes_.back()));
		auto diff = eteq::to_node<PybindT>(trainy_) - final_out;
		auto l2_regularized = tenncor::matmul(tenncor::transpose(
			sample_pipes_.back()), diff) - l2_reg * eteq::to_node<PybindT>(w);

		auto wshape = w->shape();
		auto bshape = b->shape();
		auto tlr_placeholder = eteq::make_variable_scalar<PybindT>(
			train_lr, teq::Shape(), "learning_rate");
		auto dw = eteq::to_node<PybindT>(w) +
			tenncor::extend(eteq::to_node<PybindT>(tlr_placeholder), 0,
				std::vector<teq::DimT>(wshape.begin(), wshape.end())) *
			l2_regularized;
		auto db = eteq::to_node<PybindT>(b) +
			tenncor::extend(eteq::to_node<PybindT>(tlr_placeholder), 0,
				std::vector<teq::DimT>(bshape.begin(), bshape.end())) *
			tenncor::reduce_mean_1d(diff, 1);
		auto dtrain_lr = eteq::to_node<PybindT>(tlr_placeholder) * lr_scaling;

		tupdates_ = {
			layr::AssignsT{
				layr::VarAssign{"logw_grad",
					std::static_pointer_cast<eteq::Variable<PybindT>>(w), dw},
				layr::VarAssign{"logb_grad",
					std::static_pointer_cast<eteq::Variable<PybindT>>(b), db}},
			layr::AssignsT{layr::VarAssign{"loglr_grad", tlr_placeholder, dtrain_lr}},
		};
		tcost_ = -tenncor::reduce_mean(tenncor::reduce_sum_1d(
			eteq::to_node<PybindT>(trainy_) * tenncor::log(final_out) +
			((PybindT) 1 - eteq::to_node<PybindT>(trainy_)) * tenncor::log((PybindT) 1 - final_out), 0));

		train_sess_.track({
			sample_pipes_.back()->get_tensor(),
			dw->get_tensor(),
			db->get_tensor(),
			dtrain_lr->get_tensor(),
			tcost_->get_tensor()
		});
	}

	void pretrain (teq::ShapedArr<PybindT>& train_in,
		size_t nepochs = 100,
		std::function<void(size_t,size_t)> logger = std::function<void(size_t,size_t)>())
	{
		trainx_->assign(train_in);

		for (size_t i = 0; i < nlayers_; ++i)
		{
			// train rbm layers (reconstruction) setup
			auto to_ignore = sample_pipes_[i]->get_tensor().get();
			auto& updates = rupdates_[i];

			for (size_t epoch = 0; epoch < nepochs; ++epoch)
			{
				// train rbm layers (reconstruction)
				layr::assign_groups_preupdate({updates},
					[&](teq::TensSetT& sources)
					{
						pretrain_sess_.update_target(sources, {to_ignore});
					});

				if (logger)
				{
					logger(epoch, i);
				}
			}

			if (i < nlayers_ - 1)
			{
				pretrain_sess_.update_target(
					{sample_pipes_[i + 1]->get_tensor().get()},
					{to_ignore});
			}
		}
	}

	void finetune (
		teq::ShapedArr<PybindT>& train_in,
		teq::ShapedArr<PybindT>& train_out,
		size_t nepochs = 100,
		std::function<void(size_t)> logger = std::function<void(size_t)>())
	{
		trainx_->assign(train_in);
		trainy_->assign(train_out);

		// assert len(self.sample_pipes) > 1, since self.n_layers > 0
		auto to_ignore = sample_pipes_.back()->get_tensor().get();
		auto prev_ignore = sample_pipes_[nlayers_ - 1]->get_tensor().get();
		assert(false == static_cast<eteq::Functor<PybindT>*>(prev_ignore)->is_uninit());
		train_sess_.update_target({to_ignore}, {prev_ignore});

		for (size_t epoch = 0; epoch < nepochs; ++epoch)
		{
			// train log layer
			layr::assign_groups_preupdate(tupdates_,
				[&](teq::TensSetT& sources)
				{
					train_sess_.update_target(sources, {to_ignore});
				});

			if (logger)
			{
				logger(epoch);
			}
		}
	}

	PybindT reconstruction_cost (size_t layer)
	{
		auto rcost = rcosts_[layer];
		pretrain_sess_.update_target(
			{rcost->get_tensor().get()},
			{sample_pipes_[layer]->get_tensor().get()});
		return *rcost->data();
	}

	PybindT training_cost (void)
	{
		train_sess_.update_target(
			{tcost_->get_tensor().get()},
			{sample_pipes_.back()->get_tensor().get()});
		return *tcost_->data();
	}

	size_t nlayers_;

	size_t input_size_;

	size_t output_size_;

	size_t batch_size_;

	eteq::VarptrT<PybindT> trainx_;

	eteq::VarptrT<PybindT> trainy_;

	std::vector<LinkptrT> sample_pipes_;

	layr::AssignGroupsT rupdates_;

	layr::AssignGroupsT tupdates_;

	std::vector<LinkptrT> rcosts_;

	LinkptrT tcost_;

	teq::Session pretrain_sess_;

	teq::Session train_sess_;
};

}

#endif // RCN_DBN_TRAINER_HPP
