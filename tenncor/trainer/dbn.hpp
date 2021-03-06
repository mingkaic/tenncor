
#ifndef DBN_TRAINER_HPP
#define DBN_TRAINER_HPP

#include "tenncor/trainer/rbm.hpp"
#include "tenncor/trainer/shaped_arr.hpp"

namespace trainer
{

// source: https://gist.github.com/yusugomori/4509019
// todo: add optimization options
template <typename T>
struct DBNTrainer final
{
	DBNTrainer (const std::vector<layr::RBMLayer<T>>& rbms, eteq::ETensor dense,
		teq::RankT softmax_dim, teq::DimT batch_size,
		T pretrain_lr = 0.1, T train_lr = 0.1,
		size_t cdk = 10, T l2_reg = 0., T lr_scaling = 0.95,
		global::CfgMapptrT context = global::context()) :
		nlayers_(rbms.size()),
		batch_size_(batch_size),
		context_(context)
	{
		input_size_ = layr::get_input(rbms.front().fwd_)->shape().at(0);
		output_size_ = dense->shape().at(0);

		teq::Shape inshape({(teq::DimT) input_size_, batch_size});
		teq::Shape outshape({(teq::DimT) output_size_, batch_size});
		trainx_ = eteq::make_variable_scalar<T>(0, inshape, "trainx", context);
		trainy_ = eteq::make_variable_scalar<T>(0, outshape, "trainy", context);

		// setups:
		// general rbm sampling
		sample_pipes_.push_back(trainx_);
		for (size_t i = 0; i < nlayers_; ++i)
		{
			sample_pipes_.push_back(sample_v2h(rbms[i], sample_pipes_[i]));
		}

		TenncorAPI api(context);

		// layer-wise rbm reconstruction
		teq::TensptrSetT to_track;
		for (size_t i = 0; i < nlayers_; ++i)
		{
			auto& rbm = rbms[i];
			auto& rx = sample_pipes_[i];
			auto& ry = sample_pipes_[i + 1];
			teq::TensSetT to_learn;
			auto fstorage = layr::get_storage<T>(rbm.fwd_);
			auto bstorage = layr::get_storage<T>(rbm.bwd_);
			for (auto var : fstorage)
			{
				to_learn.emplace(var.get());
			}
			for (auto var : bstorage)
			{
				to_learn.emplace(var.get());
			}

			CDChainIO<T> io(rx, ry);
			layr::VarErrsT<T> varerrs = cd_grad_approx<T>(
				io, rbm, cdk, nullptr, context); // todo: add persistent option
			teq::TensptrSetT assigns;
			for (auto varerr : varerrs)
			{
				// if var is a weight or bias add assign with learning rate
				// otherwise assign directly
				auto assign = estd::has(to_learn, varerr.first.get()) ?
					api.assign_add(eteq::EVariable<T>(varerr.first, context),
						pretrain_lr * varerr.second) :
					api.assign(eteq::EVariable<T>(varerr.first, context), varerr.second);
				assigns.emplace(assign);
				to_track.emplace(assign);
			}
			rupdates_.push_back(assigns);

			auto vhv = api.sigmoid(rbm.backward_connect(
				api.sigmoid(rbm.connect(rx))));
			auto rcost = -api.reduce_mean(api.reduce_sum_1d(
				rx * api.log(vhv) + ((T) 1 - rx) * api.log((T) 1 - vhv), 0));
			rcosts_.push_back(rcost);
			to_track.emplace(rcost);
		}
		to_track.emplace(sample_pipes_.back());

		// logistic layer training
		// todo: improve this adhoc way of training log layer
		auto contents = layr::get_storage<T>(dense);
		eteq::VarptrT<T> w = contents[0];
		eteq::VarptrT<T> b = contents[1];
		auto final_out = api.softmax(layr::connect(
			dense, sample_pipes_.back()), softmax_dim, 1);
		auto diff = trainy_ - final_out;
		auto l2_regularized = api.matmul(api.transpose(
			sample_pipes_.back()), diff) - l2_reg * eteq::ETensor(w, context);

		auto wshape = w->shape();
		auto bshape = b->shape();
		auto tlr_placeholder = eteq::make_variable_scalar<T>(
			train_lr, teq::Shape(), "learning_rate", context);
		auto dw =
			api.extend(tlr_placeholder, 0,
				teq::DimsT(wshape.begin(), wshape.end())) *
			l2_regularized;
		auto db =
			api.extend(tlr_placeholder, 0,
				teq::DimsT(bshape.begin(), bshape.end())) *
			api.reduce_mean_1d(diff, 1);
		auto dtrain_lr = tlr_placeholder * lr_scaling;

		tupdate_ = api.assign(tlr_placeholder, api.identity(dtrain_lr, {
			api.assign_add(eteq::EVariable<T>(w, context), dw),
			api.assign_add(eteq::EVariable<T>(b, context), db),
		}));
		tcost_ = -api.reduce_mean(
			api.reduce_sum_1d(trainy_ * api.log(final_out) +
			((T) 1 - trainy_) * api.log((T) 1 - final_out), 0));
	}

	void pretrain (trainer::ShapedArr<T>& train_in, size_t nepochs = 100,
		std::function<void(size_t,size_t)> logger =
			std::function<void(size_t,size_t)>())
	{
		trainx_->assign(train_in.data_.data(), train_in.shape_, context_);

		eigen::Device device;
		auto& eval = teq::get_eval(global::context());
		for (size_t i = 0; i < nlayers_; ++i)
		{
			if (auto igfnc = dynamic_cast<eteq::Functor<T>*>(sample_pipes_[i].get()))
			{
				igfnc->cache_init();
			}
		}
		for (size_t i = 0; i < nlayers_; ++i)
		{
			// train rbm layers (reconstruction) setup
			auto to_ignore = sample_pipes_[i].get();
			auto& updates = rupdates_[i];

			for (size_t epoch = 0; epoch < nepochs; ++epoch)
			{
				// train rbm layers (reconstruction)
				teq::TensSetT utens;
				utens.reserve(updates.size());
				std::transform(updates.begin(), updates.end(),
					std::inserter(utens, utens.end()),
					[](teq::TensptrT tens) { return tens.get(); });
				eval.evaluate(device, utens, {to_ignore});
				if (logger)
				{
					logger(epoch, i);
				}
			}

			if (i < nlayers_ - 1)
			{
				eval.evaluate(device,
					{sample_pipes_[i + 1].get()}, {to_ignore});
			}
		}
	}

	void finetune (trainer::ShapedArr<T>& train_in, trainer::ShapedArr<T>& train_out,
		size_t nepochs = 100, std::function<void(size_t)> logger =
			std::function<void(size_t)>())
	{
		trainx_->assign(train_in.data_.data(), train_in.shape_, context_);
		trainy_->assign(train_out.data_.data(), train_out.shape_, context_);

		eigen::Device device;
		// assert len(self.sample_pipes) > 1, since self.n_layers > 0
		auto to_ignore = sample_pipes_.back();
		auto prev_ignore = sample_pipes_[nlayers_ - 1].get();
		assert(static_cast<eteq::Functor<T>*>(prev_ignore)->has_data());
		if (auto igfnc = dynamic_cast<eteq::Functor<T>*>(to_ignore.get()))
		{
			igfnc->cache_init();
		}
		to_ignore.template calc<T>({prev_ignore});

		for (size_t epoch = 0; epoch < nepochs; ++epoch)
		{
			// train log layer
			tupdate_.template calc<T>({to_ignore.get()});
			if (logger)
			{
				logger(epoch);
			}
		}
	}

	T reconstruction_cost (size_t layer)
	{
		auto& rcost = rcosts_[layer];
		return *rcost.template calc<T>();
	}

	T training_cost (void)
	{
		return *tcost_.template calc<T>();
	}

	size_t nlayers_;

	size_t input_size_;

	size_t output_size_;

	size_t batch_size_;

	eteq::EVariable<T> trainx_;

	eteq::EVariable<T> trainy_;

	eteq::ETensorsT sample_pipes_;

	std::vector<teq::TensptrSetT> rupdates_;

	eteq::ETensor tupdate_;

	eteq::ETensorsT rcosts_;

	eteq::ETensor tcost_;

	global::CfgMapptrT context_;
};

}

#endif // DBN_TRAINER_HPP
