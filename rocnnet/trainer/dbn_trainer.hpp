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
		rbm_ltype != layr::rbm_layer_key) ||
		activation_ltype != layr::softmax_layer_key)
	{
		return false;
	}
	return true;
}

eteq::NodeptrT<PybindT> sample_h_given_v (
	layr::RBMptrT& rbm, eteq::NodeptrT<PybindT>& x)
{
	return tenncor::random::rand_binom_one(rbm->connect(x));
}

eteq::NodeptrT<PybindT> sample_v_given_h (
	layr::RBMptrT& rbm, eteq::NodeptrT<PybindT>& x)
{
	return tenncor::random::rand_binom_one(rbm->backward_connect(x));
}

// implementation taken from https://gist.github.com/yusugomori/4509019
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
		sample_pipes_.push_back(convert_to_node(trainx_));
		for (size_t i = 0; i < nlayers_; ++i)
		{
			sample_pipes_.push_back(sample_h_given_v(
				rbm_layers[i], sample_pipes_[i]));
		}

		// layer-wise rbm reconstruction
		teq::TensT to_track;
		for (size_t i = 0; i < nlayers_; ++i)
		{
			auto& rbm = rbm_layers[i];
			auto& rx = sample_pipes_[i];
			auto& ry = sample_pipes_[i + 1];

			auto contents = rbm->get_contents();
			auto& w = contents[0];
			auto& hb = contents[1];
			auto& vb = contents[3];
			auto chain_it = ry;
			for (size_t i = 0; i < cdk - 1; ++i)
			{
				chain_it = tenncor::random::rand_binom_one(rbm->connect(
					sample_v_given_h(rbm, chain_it)));
			}
			auto nv_samples = sample_v_given_h(rbm, chain_it);
			auto nh_means = rbm->connect(nv_samples);

			auto dw = eteq::NodeConverters<PybindT>::to_node(w) + pretrain_lr * (
				tenncor::matmul(tenncor::transpose(rx), ry) -
				tenncor::matmul(tenncor::transpose(nv_samples), nh_means));
			auto dhb = eteq::NodeConverters<PybindT>::to_node(hb) + pretrain_lr *
				tenncor::reduce_mean_1d(ry - nh_means, 1);
			auto dvb = eteq::NodeConverters<PybindT>::to_node(vb) + pretrain_lr *
				tenncor::reduce_mean_1d(rx - nv_samples, 1);

			rupdates_.push_back(layr::AssignsT{
				layr::VarAssign{
					fmts::sprintf("rbm_dw_%d", i),
					std::make_shared<eteq::VariableNode<PybindT>>(
						std::static_pointer_cast<eteq::Variable<PybindT>>(w)),
					dw},
				layr::VarAssign{
					fmts::sprintf("rbm_dhb_%d", i),
					std::make_shared<eteq::VariableNode<PybindT>>(
						std::static_pointer_cast<eteq::Variable<PybindT>>(hb)),
					dhb},
				layr::VarAssign{
					fmts::sprintf("rbm_dvb_%d", i),
					std::make_shared<eteq::VariableNode<PybindT>>(
						std::static_pointer_cast<eteq::Variable<PybindT>>(vb)),
					dvb},
			});

			auto vhv = rbm->backward_connect(rbm->connect(rx));
			auto rcost = -tenncor::reduce_mean(
				tenncor::reduce_sum_1d(rx * tenncor::log(vhv) +
					(1.f - rx) * tenncor::log(1.f - vhv), 0));
			rcosts_.push_back(rcost);

			to_track.push_back(dw->get_tensor());
			to_track.push_back(dhb->get_tensor());
			to_track.push_back(dvb->get_tensor());
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
		auto diff = eteq::convert_to_node(trainy_) - final_out;
		auto l2_regularized = tenncor::matmul(tenncor::transpose(
			sample_pipes_.back()), diff) -
			l2_reg * eteq::NodeConverters<PybindT>::to_node(w);

		auto wshape = w->shape();
		auto bshape = b->shape();
		auto tlr_placeholder = eteq::make_variable_scalar<PybindT>(
			train_lr, teq::Shape(), "learning_rate");
		auto dw = eteq::NodeConverters<PybindT>::to_node(w) +
			tenncor::extend(eteq::convert_to_node(tlr_placeholder), 0,
				std::vector<teq::DimT>(wshape.begin(), wshape.end())) *
			l2_regularized;
		auto db = eteq::NodeConverters<PybindT>::to_node(b) +
			tenncor::extend(eteq::convert_to_node(tlr_placeholder), 0,
				std::vector<teq::DimT>(bshape.begin(), bshape.end())) *
			tenncor::reduce_mean_1d(diff, 1);
		auto dtrain_lr = eteq::convert_to_node(tlr_placeholder) * lr_scaling;

		tupdates_ = {
			layr::AssignsT{
				layr::VarAssign{"logw_grad",
					std::make_shared<eteq::VariableNode<PybindT>>(
						std::static_pointer_cast<eteq::Variable<PybindT>>(w)), dw},
				layr::VarAssign{"logb_grad",
					std::make_shared<eteq::VariableNode<PybindT>>(
						std::static_pointer_cast<eteq::Variable<PybindT>>(b)), db}},
			layr::AssignsT{layr::VarAssign{"loglr_grad", tlr_placeholder, dtrain_lr}},
		};
		tcost_ = -tenncor::reduce_mean(tenncor::reduce_sum_1d(
			eteq::convert_to_node(trainy_) * tenncor::log(final_out) +
			(1.f - eteq::convert_to_node(trainy_)) * tenncor::log(1.f - final_out), 0));

		train_sess_.track({
			sample_pipes_.back()->get_tensor(),
			dw->get_tensor(),
			db->get_tensor(),
			dtrain_lr->get_tensor(),
			tcost_->get_tensor()
		});
	}

	void pretrain (std::vector<PybindT>& train_in,
		size_t nepochs = 100,
		std::function<void(size_t,size_t)> logger = std::function<void(size_t,size_t)>())
	{
		if (train_in.size() != input_size_ * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match "
				"input size (%d) * batchsize (%d)", train_in.size(),
				input_size_, batch_size_);
		}
		trainx_->assign(train_in.data(), trainx_->shape());

		for (size_t i = 0; i < nlayers_; ++i)
		{
			// train rbm layers (reconstruction) setup
			auto to_ignore = sample_pipes_[i]->get_tensor().get();
			auto& updates = rupdates_[i];

			for (size_t epoch = 0; epoch < nepochs; ++epoch)
			{
				// train rbm layers (reconstruction)
				layr::assign_groups_preupdate({updates},
					[&](eteq::TensSetT& sources)
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
		std::vector<PybindT>& train_in,
		std::vector<PybindT>& train_out,
		size_t nepochs = 100,
		std::function<void(size_t)> logger = std::function<void(size_t)>())
	{
		if (train_in.size() != input_size_ * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match "
				"input size (%d) * batchsize (%d)", train_in.size(),
				input_size_, batch_size_);
		}
		if (train_out.size() != output_size_ * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match "
				"output size (%d) * batchsize (%d)", train_in.size(),
				output_size_, batch_size_);
		}
		trainx_->assign(train_in.data(), trainx_->shape());
		trainy_->assign(train_out.data(), trainy_->shape());

		// assert len(self.sample_pipes) > 1, since self.n_layers > 0
		auto to_ignore = sample_pipes_.back()->get_tensor().get();
		auto prev_ignore = sample_pipes_[nlayers_ - 1]->get_tensor().get();
		if (static_cast<eteq::Functor<PybindT>*>(prev_ignore)->is_uninit())
		{
			train_sess_.update_target({to_ignore}); // update everything
		}
		else
		{
			train_sess_.update_target({to_ignore},
				{prev_ignore});
		}

		for (size_t epoch = 0; epoch < nepochs; ++epoch)
		{
			// train log layer
			layr::assign_groups_preupdate(tupdates_,
				[&](eteq::TensSetT& sources)
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

	std::vector<eteq::NodeptrT<PybindT>> sample_pipes_;

	layr::AssignGroupsT rupdates_;

	layr::AssignGroupsT tupdates_;

	std::vector<eteq::NodeptrT<PybindT>> rcosts_;

	eteq::NodeptrT<PybindT> tcost_;

	eteq::Session pretrain_sess_;

	eteq::Session train_sess_;
};

}

#endif // RCN_DBN_TRAINER_HPP
