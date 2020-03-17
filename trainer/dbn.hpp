#include "trainer/rbm.hpp"

#ifndef DBN_TRAINER_HPP
#define DBN_TRAINER_HPP

namespace trainer
{

// source: https://gist.github.com/yusugomori/4509019
// todo: add optimization options
template <typename T>
struct DBNTrainer final
{
	DBNTrainer (const std::vector<layr::RBMLayer<T>>& rbms,
		eteq::ETensor<T> dense, teq::RankT softmax_dim,
		teq::DimT batch_size, T pretrain_lr = 0.1,
		T train_lr = 0.1, size_t cdk = 10,
		T l2_reg = 0., T lr_scaling = 0.95) :
		nlayers_(rbms.size()),
		batch_size_(batch_size),
		pretrain_sess_(eigen::default_device()),
		train_sess_(eigen::default_device())
	{
		input_size_ = eteq::get_input(layr::dense_name, rbms.front().fwd_)->shape().at(0);
		output_size_ = dense->shape().at(0);

		teq::Shape inshape({(teq::DimT) input_size_, batch_size});
		teq::Shape outshape({(teq::DimT) output_size_, batch_size});
		trainx_ = eteq::make_variable_scalar<T>(0, inshape, "trainx");
		trainy_ = eteq::make_variable_scalar<T>(0, outshape, "trainy");

		// setups:
		// general rbm sampling
		sample_pipes_.push_back(trainx_);
		for (size_t i = 0; i < nlayers_; ++i)
		{
			sample_pipes_.push_back(sample_v2h(rbms[i], sample_pipes_[i]));
		}

		// layer-wise rbm reconstruction
		teq::TensptrsT to_track;
		for (size_t i = 0; i < nlayers_; ++i)
		{
			auto& rbm = rbms[i];
			auto& rx = sample_pipes_[i];
			auto& ry = sample_pipes_[i + 1];
			teq::TensSetT to_learn;
			auto fstorage = eteq::get_storage(layr::dense_name, rbm.fwd_);
			auto bstorage = eteq::get_storage(layr::dense_name, rbm.bwd_);
			for (auto var : fstorage)
			{
				to_learn.emplace(var.get());
			}
			for (auto var : bstorage)
			{
				to_learn.emplace(var.get());
			}

			CDChainIO io(rx, ry);
			layr::VarMapT<T> varerrs = cd_grad_approx(io, rbm, cdk); // todo: add persistent option
			teq::TensSetT assigns;
			for (auto varerr : varerrs)
			{
				// if var is a weight or bias add assign with learning rate
				// otherwise assign directly
				auto assign = estd::has(to_learn, varerr.first.get()) ?
					tenncor::assign_add(eteq::EVariable<T>(varerr.first), pretrain_lr * varerr.second) :
					tenncor::assign(eteq::EVariable<T>(varerr.first), varerr.second);
				assigns.emplace(assign.get());
				to_track.push_back(assign);
			}
			rupdates_.push_back(assigns);

			auto vhv = tenncor::sigmoid(rbm.backward_connect(
				tenncor::sigmoid(rbm.connect(rx))));
			auto rcost = -tenncor::reduce_mean(tenncor::reduce_sum_1d(
				rx * tenncor::log(vhv) + ((T) 1 - rx) *
				tenncor::log((T) 1 - vhv), 0));
			rcosts_.push_back(rcost);
			to_track.push_back(rcost);
		}
		to_track.push_back(sample_pipes_.back());
		pretrain_sess_.track(to_track);

		// logistic layer training
		// todo: improve this adhoc way of training log layer
		auto contents = eteq::get_storage(layr::dense_name, dense);
		eteq::VarptrT<T> w = contents[0];
		eteq::VarptrT<T> b = contents[1];
		auto final_out = tenncor::softmax(eteq::connect(layr::dense_name, dense, sample_pipes_.back()), softmax_dim, 1);
		auto diff = trainy_ - final_out;
		auto l2_regularized = tenncor::matmul(tenncor::transpose(
			sample_pipes_.back()), diff) - l2_reg * eteq::ETensor<T>(w);

		auto wshape = w->shape();
		auto bshape = b->shape();
		auto tlr_placeholder = eteq::make_variable_scalar<T>(
			train_lr, teq::Shape(), "learning_rate");
		auto dw =
			tenncor::extend(tlr_placeholder, 0,
				std::vector<teq::DimT>(wshape.begin(), wshape.end())) *
			l2_regularized;
		auto db =
			tenncor::extend(tlr_placeholder, 0,
				std::vector<teq::DimT>(bshape.begin(), bshape.end())) *
			tenncor::reduce_mean_1d(diff, 1);
		auto dtrain_lr = tlr_placeholder * lr_scaling;

		tupdate_ = tenncor::depends(tenncor::assign(tlr_placeholder, dtrain_lr),
		eteq::ETensorsT<T>{
			tenncor::assign_add(eteq::EVariable<T>(w), dw),
			tenncor::assign_add(eteq::EVariable<T>(b), db),
		});
		tcost_ = -tenncor::reduce_mean(
			tenncor::reduce_sum_1d(trainy_ * tenncor::log(final_out) +
			((T) 1 - trainy_) * tenncor::log((T) 1 - final_out), 0));

		train_sess_.track({
			(teq::TensptrT) sample_pipes_.back(),
			(teq::TensptrT) tupdate_,
			(teq::TensptrT) tcost_,
		});
	}

	void pretrain (teq::ShapedArr<T>& train_in, size_t nepochs = 100,
		std::function<void(size_t,size_t)> logger =
			std::function<void(size_t,size_t)>())
	{
		trainx_->assign(train_in);

		for (size_t i = 0; i < nlayers_; ++i)
		{
			// train rbm layers (reconstruction) setup
			auto to_ignore = sample_pipes_[i].get();
			auto& updates = rupdates_[i];

			for (size_t epoch = 0; epoch < nepochs; ++epoch)
			{
				// train rbm layers (reconstruction)
				pretrain_sess_.update_target(updates, {to_ignore});
				if (logger)
				{
					logger(epoch, i);
				}
			}

			if (i < nlayers_ - 1)
			{
				pretrain_sess_.update_target(
					{sample_pipes_[i + 1].get()}, {to_ignore});
			}
		}
	}

	void finetune (teq::ShapedArr<T>& train_in, teq::ShapedArr<T>& train_out,
		size_t nepochs = 100, std::function<void(size_t)> logger =
			std::function<void(size_t)>())
	{
		trainx_->assign(train_in);
		trainy_->assign(train_out);

		// assert len(self.sample_pipes) > 1, since self.n_layers > 0
		auto to_ignore = sample_pipes_.back().get();
		auto prev_ignore = sample_pipes_[nlayers_ - 1].get();
		assert(static_cast<eteq::Functor<T>*>(prev_ignore)->has_data());
		train_sess_.update_target({to_ignore}, {prev_ignore});

		for (size_t epoch = 0; epoch < nepochs; ++epoch)
		{
			// train log layer
			train_sess_.update_target({tupdate_.get()}, {to_ignore});
			if (logger)
			{
				logger(epoch);
			}
		}
	}

	T reconstruction_cost (size_t layer)
	{
		auto rcost = rcosts_[layer];
		pretrain_sess_.update_target(
			{rcost.get()}, {sample_pipes_[layer].get()});
		return *((T*) rcost->device().data());
	}

	T training_cost (void)
	{
		train_sess_.update_target(
			{tcost_.get()}, {sample_pipes_.back().get()});
		return *((T*) tcost_->device().data());
	}

	size_t nlayers_;

	size_t input_size_;

	size_t output_size_;

	size_t batch_size_;

	eteq::EVariable<T> trainx_;

	eteq::EVariable<T> trainy_;

	eteq::ETensorsT<T> sample_pipes_;

	std::vector<teq::TensSetT> rupdates_;

	eteq::ETensor<T> tupdate_;

	eteq::ETensorsT<T> rcosts_;

	eteq::ETensor<T> tcost_;

	teq::Session pretrain_sess_;

	teq::Session train_sess_;
};

}

#endif // DBN_TRAINER_HPP
