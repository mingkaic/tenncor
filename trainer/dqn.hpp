#include "trainer/trainer.hpp"

#ifndef DQN_TRAINER_HPP
#define DQN_TRAINER_HPP

namespace trainer
{

// experience replay
template <typename T>
struct ExpBatch
{
	std::vector<T> observation_;
	size_t action_idx_;
	T reward_;
	std::vector<T> new_observation_;
};

template <typename T>
struct DQNTrainer final
{
	DQNTrainer (eteq::ETensor<T>& model, teq::iSession& sess,
		layr::ApproxF<T> update, layr::UnaryF<T> gradprocess = layr::UnaryF<T>(),
		size_t train_interval = 5, T rand_action_prob = 0.05,
		T discount_rate = 0.95, T target_update_rate = 0.01,
		T explore_period = 1000, size_t store_interval = 5,
		teq::DimT mbatch_size = 32, size_t max_exp = 30000) :
		source_model_(model), target_model_(eteq::deep_clone(model)), sess_(&sess),
		train_interval_(train_interval), rand_action_prob_(rand_action_prob),
		discount_rate_(discount_rate), target_update_rate_(target_update_rate),
		explore_period_(explore_period), store_interval_(store_interval),
		mbatch_size_(mbatch_size), max_exp_(max_exp),
		get_random_(eigen::Randomizer().unif_gen<T>(0, 1))
	{
		teq::DimT indim = eteq::get_input(model)->shape().at(0);
		teq::DimT outdim = model->shape().at(0);
		teq::Shape batchin({indim, mbatch_size_});
		teq::Shape batchout({outdim, mbatch_size_});
		teq::Shape outshape({mbatch_size_});

		obs_input_ = eteq::make_variable_scalar<T>(0, teq::Shape({indim}), "obs");
		src_input_ = eteq::make_variable_scalar<T>(0, batchin, "src_obs");
		nxt_input_ = eteq::make_variable_scalar<T>(0, batchin, "nxt_obs");
		src_outmask_ = eteq::make_variable_scalar<T>(1, batchout, "src_outmask");
		nxt_outmask_ = eteq::make_variable_scalar<T>(1, outshape, "nxt_outmask");
		reward_input_ = eteq::make_variable_scalar<T>(0, outshape, "rewards");

		// forward action score computation
		action_idx_ = tenncor::argmax(eteq::connect(source_model_, obs_input_));

		src_act_ = eteq::connect(source_model_, src_input_);

		// predicting target future rewards
		nxt_act_ = eteq::connect(target_model_, nxt_input_);
		auto target_vals = nxt_outmask_ * tenncor::reduce_max_1d(nxt_act_, 0);
		auto future_reward = reward_input_ + discount_rate_ * target_vals;
		auto masked_output_score = tenncor::reduce_sum_1d(src_act_ * src_outmask_, 0);
		auto prediction_err = tenncor::reduce_mean(tenncor::square(
			masked_output_score - future_reward));

		eteq::VarptrsT<T> source_vars = eteq::get_storage(source_model_);
		eteq::VarptrsT<T> target_vars = eteq::get_storage(target_model_);
		assert(source_vars.size() == target_vars.size());

		layr::VarMapT<T> source_errs;
		for (auto& source_var : source_vars)
		{
			// updates for source network
			auto error = eteq::derive(prediction_err, eteq::ETensor<T>(source_var));
			if (gradprocess)
			{
				error = gradprocess(error);
			}
			source_errs.emplace(eteq::EVariable<T>(source_var), error);
		}
		auto src_updates = update(source_errs);

		teq::TensptrsT track_batch = {prediction_err, src_act_, action_idx_};
		track_batch.reserve(source_vars.size() + 3);
		for (size_t i = 0; i < source_vars.size(); ++i)
		{
			// updates for target network
			eteq::EVariable<T> target_var = target_vars[i];
			auto diff = target_var - src_updates[source_vars[i]];
			auto assign = tenncor::assign_sub(target_var, target_update_rate_ * diff);
			track_batch.push_back(assign);
			updates_.emplace(assign.get());
		}
		sess_->track(track_batch);
	}

	uint8_t action (const teq::ShapedArr<T>& input)
	{
		actions_executed_++; // book keep
		T exploration = linear_annealing(1.);
		// perform random exploration action
		if (get_random_() < exploration)
		{
			return std::floor(get_random_() * source_model_->shape().at(0));
		}
		obs_input_->assign(input);
		sess_->update();
		return *((T*) action_idx_->device().data());
	}

	void store (std::vector<T> observation, size_t action_idx,
		T reward, std::vector<T> new_obs)
	{
		if (0 == nstore_called_ % store_interval_)
		{
			experiences_.push_back(ExpBatch<T>{
				observation, action_idx, reward, new_obs});
			if (experiences_.size() > max_exp_)
			{
				experiences_.erase(experiences_.begin());
			}
		}
		nstore_called_++;
	}

	void train (void)
	{
		if (experiences_.size() < mbatch_size_)
		{
			return;
		}
		// extract mini_batch from buffer and backpropagate
		if (0 == (ntrain_called_ % train_interval_))
		{
			std::vector<ExpBatch<T>> samples = random_sample();

			// batch data process
			std::vector<T> states; // <ninput, batchsize>
			std::vector<T> new_states; // <ninput, batchsize>
			std::vector<T> action_mask; // <noutput, batchsize>
			std::vector<T> rewards; // <batchsize>

			for (size_t i = 0, n = samples.size(); i < n; i++)
			{
				ExpBatch<T> batch = samples[i];
				auto& observation = batch.observation_;
				auto& action_idx = batch.action_idx_;
				auto& reward = batch.reward_;
				auto& new_obs = batch.new_observation_;
				assert(new_obs.size() > 0);

				states.insert(states.end(), observation.begin(), observation.end());
				std::vector<T> local_act_mask(source_model_->shape().at(0), 0);
				local_act_mask[action_idx] = 1.;
				action_mask.insert(action_mask.end(), local_act_mask.begin(), local_act_mask.end());
				rewards.push_back(reward);
				new_states.insert(new_states.end(), new_obs.begin(), new_obs.end());
			}

			// enter processed batch data
			src_input_->assign(states.data(), src_input_->shape());
			src_outmask_->assign(action_mask.data(), src_outmask_->shape());
			nxt_input_->assign(new_states.data(), nxt_input_->shape());
			reward_input_->assign(rewards.data(), reward_input_->shape());

			sess_->update_target(updates_);
		}
		ntrain_called_++;
	}

	// === forward computation ===
	// fanin: shape <ninput>
	eteq::EVariable<T> obs_input_;

	// fanout: shape <1>
	eteq::ETensor<T> action_idx_;

	// === backward computation ===
	// train fanin: shape <ninput, batchsize>
	eteq::EVariable<T> src_input_;

	// train fanout: shape <noutput, batchsize>
	eteq::ETensor<T> src_act_;

	// === updates && optimizer ===
	teq::TensSetT  updates_;

	teq::iSession* sess_;

private:
	T linear_annealing (T initial_prob) const
	{
		if (actions_executed_ >= explore_period_)
		{
			return rand_action_prob_;
		}
		return initial_prob - actions_executed_ * (initial_prob -
			rand_action_prob_) / explore_period_;
	}

	std::vector<ExpBatch<T>> random_sample (void)
	{
		std::vector<size_t> indices(experiences_.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), eigen::default_engine());
		std::vector<ExpBatch<T>> res;
		for (size_t i = 0; i < mbatch_size_; i++)
		{
			size_t idx = indices[i];
			res.push_back(experiences_[idx]);
		}
		return res;
	}

	// === scalar parameters ===
	// training parameters
	size_t train_interval_ = 5;
	T rand_action_prob_ = 0.05;
	T discount_rate_ = 0.95;
	T target_update_rate_ = 0.01;
	T explore_period_ = 1000;
	// memory parameters
	size_t store_interval_ = 5;
	teq::DimT mbatch_size_ = 32;
	size_t max_exp_ = 30000;
	// non-input parameters
	size_t actions_executed_ = 0;
	size_t ntrain_called_ = 0;
	size_t nstore_called_ = 0;
	std::vector<ExpBatch<T>> experiences_;

	// source network
	eteq::ETensor<T> source_model_;

	// target network
	eteq::ETensor<T> target_model_;

	// train fanout: shape <noutput, batchsize>
	eteq::ETensor<T> nxt_act_;

	// === prediction computation ===
	// train_fanin: shape <ninput, batchsize>
	eteq::EVariable<T> nxt_input_;

	// train mask: shape <batchsize>
	eteq::EVariable<T> nxt_outmask_;

	// reward associated with nxt_act_: shape <batchsize>
	eteq::EVariable<T> reward_input_;

	// === q-value computation ===
	// weight output to get overall score: shape <noutput, batchsize>
	eteq::EVariable<T> src_outmask_;

	// overall score: shape <noutput>
	eteq::ETensor<T> score_;

	// states
	eigen::GenF<T> get_random_;
};

}

#endif // DQN_TRAINER_HPP
