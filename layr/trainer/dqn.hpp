#include "layr/trainer/trainer.hpp"

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
struct DQNInfo final
{
	DQNInfo (size_t train_interval = 5,
		T rand_action_prob = 0.05,
		T discount_rate = 0.95,
		T target_update_rate = 0.01,
		T exploration_period = 1000,
		size_t store_interval = 5,
		teq::DimT mini_batch_size = 32,
		size_t max_exp = 30000) :
		train_interval_(train_interval),
		rand_action_prob_(rand_action_prob),
		discount_rate_(discount_rate),
		target_update_rate_(target_update_rate),
		exploration_period_(exploration_period),
		store_interval_(store_interval),
		mini_batch_size_(mini_batch_size),
		max_exp_(max_exp) {}

	size_t train_interval_ = 5;
	T rand_action_prob_ = 0.05;
	T discount_rate_ = 0.95;
	T target_update_rate_ = 0.01;
	T exploration_period_ = 1000;
	// memory parameters
	size_t store_interval_ = 5;
	teq::DimT mini_batch_size_ = 32;
	size_t max_exp_ = 30000;

	// uninit params
	size_t actions_executed_ = 0;
	size_t iteration_ = 0;
	size_t ntrain_called_ = 0;
	size_t nstore_called_ = 0;
	std::vector<ExpBatch<T>> experiences_;
};

template <typename T>
struct DQNTrainer final
{
	DQNTrainer (eteq::ELayer<T>& model, teq::iSession& sess,
		layr::ApproxF<T> update, const DQNInfo<T>& param,
		layr::UnaryF<T> gradprocess = layr::UnaryF<T>()) :
		sess_(&sess), params_(param),
		source_model_(model), target_model_(model.deep_clone()),
		get_random_(eigen::Randomizer().unif_gen<T>(0, 1))
	{
		teq::DimT indim = model.input()->shape().at(0);
		teq::DimT outdim = model.root()->shape().at(0);
		teq::Shape batchin({indim, params_.mini_batch_size_});
		teq::Shape batchout({outdim, params_.mini_batch_size_});
		teq::Shape outshape({params_.mini_batch_size_});

		input_ = eteq::make_variable_scalar<T>(0, teq::Shape({indim}), "observation");
		train_input_ = eteq::make_variable_scalar<T>(0, batchin, "train_observation");
		next_input_ = eteq::make_variable_scalar<T>(0, batchin, "next_observation");
		next_output_mask_ = eteq::make_variable_scalar<T>(0, outshape, "next_observation_mask");
		reward_ = eteq::make_variable_scalar<T>(0, outshape, "rewards");
		output_mask_ = eteq::make_variable_scalar<T>(0, batchout, "action_mask");

		// forward action score computation
		output_ = source_model_.connect(input_);
		train_out_ = source_model_.connect(train_input_);

		// predicting target future rewards
		next_output_ = target_model_.connect(next_input_);

		auto target_values = next_output_mask_ *
			tenncor::reduce_max_1d(next_output_, 0);
		// reward for each instance in batch
		future_reward_ = reward_ + params_.discount_rate_ * target_values;

		// prediction error
		auto masked_output_score = tenncor::reduce_sum_1d(
			train_out_ * output_mask_, 0);
		prediction_error_ = tenncor::reduce_mean(tenncor::square(
			masked_output_score - future_reward_));

		eteq::VarptrsT<T> source_vars = source_model_.get_storage();
		eteq::VarptrsT<T> target_vars = target_model_.get_storage();
		size_t nvars = source_vars.size();
		assert(nvars == target_vars.size());

		layr::VarMapT<T> source_errs;
		for (size_t i = 0; i < nvars; ++i)
		{
			// updates for source network
			auto source_var = source_vars[i];
			auto error = eteq::derive(
				prediction_error_, eteq::ETensor<T>(source_var));
			if (gradprocess)
			{
				error = gradprocess(error);
			}
			source_errs.emplace(eteq::EVariable<T>(source_var), error);
		}
		auto src_updates = update(source_errs);

		teq::TensptrsT track_batch = {prediction_error_, train_out_, output_};
		track_batch.reserve(nvars + 3);
		for (size_t i = 0; i < nvars; ++i)
		{
			// updates for target network
			eteq::EVariable<T> target_var = target_vars[i];
			auto diff = target_var - src_updates[source_vars[i]];
			auto assign = tenncor::assign_sub(target_var,
				params_.target_update_rate_ * diff);
			track_batch.push_back(assign);
			updates_.emplace(assign.get());
		}
		sess_->track(track_batch);
	}

	uint8_t action (const teq::ShapedArr<T>& input)
	{
		params_.actions_executed_++; // book keep
		T exploration = linear_annealing(1.);
		// perform random exploration action
		if (get_random_() < exploration)
		{
			return std::floor(get_random_() * source_model_.root()->shape().at(0));
		}
		input_->assign(input);
		sess_->update();
		T* dptr = (T*) output_->data();
		uint8_t max_i = 0;
		for (uint8_t i = 1, n = output_->shape().n_elems(); i < n; ++i)
		{
			if (dptr[max_i] < dptr[i])
			{
				max_i = i;
			}
		}
		return max_i;
	}

	void store (std::vector<T> observation, size_t action_idx,
		T reward, std::vector<T> new_obs)
	{
		if (0 == params_.nstore_called_ % params_.store_interval_)
		{
			params_.experiences_.push_back(ExpBatch<T>{
				observation, action_idx, reward, new_obs});
			if (params_.experiences_.size() > params_.max_exp_)
			{
				params_.experiences_.front() =
					std::move(params_.experiences_.back());
				params_.experiences_.pop_back();
			}
		}
		params_.nstore_called_++;
	}

	void train (void)
	{
		// extract mini_batch from buffer and backpropagate
		if (0 == (params_.ntrain_called_ % params_.train_interval_))
		{
			if (params_.experiences_.size() < params_.mini_batch_size_)
			{
				return;
			}

			std::vector<ExpBatch<T>> samples = random_sample();

			// batch data process
			std::vector<T> states; // <ninput, batchsize>
			std::vector<T> new_states; // <ninput, batchsize>
			std::vector<T> action_mask; // <noutput, batchsize>
			std::vector<T> new_states_mask; // <batchsize>
			std::vector<T> rewards; // <batchsize>

			for (size_t i = 0, n = samples.size(); i < n; i++)
			{
				ExpBatch<T> batch = samples[i];
				states.insert(states.end(),
					batch.observation_.begin(), batch.observation_.end());
				{
					std::vector<T> local_act_mask(
						source_model_.root()->shape().at(0), 0);
					local_act_mask[batch.action_idx_] = 1.;
					action_mask.insert(action_mask.end(),
						local_act_mask.begin(), local_act_mask.end());
				}
				rewards.push_back(batch.reward_);
				if (batch.new_observation_.empty())
				{
					new_states.insert(new_states.end(),
						source_model_.input()->shape().at(0), 0);
					new_states_mask.push_back(0);
				}
				else
				{
					new_states.insert(new_states.end(),
						batch.new_observation_.begin(),
						batch.new_observation_.end());
					new_states_mask.push_back(1);
				}
			}

			// enter processed batch data
			train_input_->assign(states.data(), train_input_->shape());
			output_mask_->assign(action_mask.data(), output_mask_->shape());
			next_input_->assign(new_states.data(), next_input_->shape());
			next_output_mask_->assign(new_states_mask.data(), next_output_mask_->shape());
			reward_->assign(rewards.data(), reward_->shape());

			sess_->update_target(updates_);
			params_.iteration_++;
		}
		params_.ntrain_called_++;
	}

	eteq::ETensor<T> get_error (void) const
	{
		return prediction_error_;
	}

	size_t get_numtrained (void) const
	{
		return params_.iteration_;
	}

	// === forward computation ===
	// fanin: shape <ninput>
	eteq::EVariable<T> input_;

	// fanout: shape <noutput>
	eteq::ETensor<T> output_;

	// === backward computation ===
	// train fanin: shape <ninput, batchsize>
	eteq::EVariable<T> train_input_;

	// train fanout: shape <noutput, batchsize>
	eteq::ETensor<T> train_out_;

	// === updates && optimizer ===
	teq::TensSetT  updates_;

	teq::iSession* sess_;

private:
	T linear_annealing (T initial_prob) const
	{
		if (params_.actions_executed_ >= params_.exploration_period_)
		{
			return params_.rand_action_prob_;
		}
		return initial_prob - params_.actions_executed_ * (initial_prob -
			params_.rand_action_prob_) / params_.exploration_period_;
	}

	std::vector<ExpBatch<T>> random_sample (void)
	{
		std::vector<size_t> indices(params_.experiences_.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), eigen::default_engine());
		std::vector<ExpBatch<T>> res;
		for (size_t i = 0; i < params_.mini_batch_size_; i++)
		{
			size_t idx = indices[i];
			res.push_back(params_.experiences_[idx]);
		}
		return res;
	}

	// === scalar parameters ===
	// training parameters
	DQNInfo<T> params_;

	// source network
	eteq::ELayer<T> source_model_;

	// target network
	eteq::ELayer<T> target_model_;

	// train fanout: shape <noutput, batchsize>
	eteq::ETensor<T> next_output_;

	// === prediction computation ===
	// train_fanin: shape <ninput, batchsize>
	eteq::EVariable<T> next_input_;

	// train mask: shape <batchsize>
	eteq::EVariable<T> next_output_mask_;

	// reward associated with next_output_: shape <batchsize>
	eteq::EVariable<T> reward_;

	// future reward calculated from reward history: <1, batchsize>
	eteq::ETensor<T> future_reward_;

	// === q-value computation ===
	// weight output to get overall score: shape <noutput, batchsize>
	eteq::EVariable<T> output_mask_;

	// overall score: shape <noutput>
	eteq::ETensor<T> score_;

	// future error that we want to minimize: scalar shape
	eteq::ETensor<T> prediction_error_;

	// states
	eigen::GenF<T> get_random_;
};

}

#endif // DQN_TRAINER_HPP
