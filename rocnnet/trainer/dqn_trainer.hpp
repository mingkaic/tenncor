#include "rocnnet/modl/mlp.hpp"

#include <fstream>
#include "dbg/ade.hpp"
#include "dbg/ade_csv.hpp"

struct DQNInfo
{
	DQNInfo (size_t train_interval = 5,
		double rand_action_prob = 0.05,
		double discount_rate = 0.95,
		double target_update_rate = 0.01,
		double exploration_period = 1000,
		size_t store_interval = 5,
		uint8_t mini_batch_size = 32,
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
	double rand_action_prob_ = 0.05;
	double discount_rate_ = 0.95;
	double target_update_rate_ = 0.01;
	double exploration_period_ = 1000;
	// memory parameters
	size_t store_interval_ = 5;
	uint8_t mini_batch_size_ = 32;
	size_t max_exp_ = 30000;
};

struct DQNTrainer
{
	DQNTrainer (modl::MLPptrT brain, ead::Session<double>& sess,
		eqns::ApproxFuncT update, DQNInfo param) :
		sess_(&sess),
		params_(param),
		source_qnet_(brain),
		target_qnet_(std::make_shared<modl::MLP>(*brain))
	{
		input_ = ead::make_variable_scalar<double>(0.0,
			ade::Shape({source_qnet_->get_ninput()}), "observation");
		train_input_ = ead::make_variable_scalar<double>(0.0,
			ade::Shape({source_qnet_->get_ninput(),
			params_.mini_batch_size_}), "train_observation");
		next_input_ = ead::make_variable_scalar<double>(0.0,
			ade::Shape({source_qnet_->get_ninput(),
			params_.mini_batch_size_}), "next_observation");
		next_output_mask_ = ead::make_variable_scalar<double>(0.0,
			ade::Shape({params_.mini_batch_size_}),
			"next_observation_mask");
		reward_ = ead::make_variable_scalar<double>(0.0,
			ade::Shape({params_.mini_batch_size_}), "rewards");
		output_mask_ = ead::make_variable_scalar<double>(0.0,
			ade::Shape({source_qnet_->get_noutput(),
			params_.mini_batch_size_}), "action_mask");

		// forward action score computation
		output_ = (*source_qnet_)(ead::convert_to_node<double>(input_));

		train_out_ = (*source_qnet_)(ead::convert_to_node<double>(train_input_));

		// predicting target future rewards
		next_output_ = (*target_qnet_)(ead::convert_to_node<double>(next_input_));

		auto target_values = age::mul(
			age::reduce_max_1d(next_output_, 0),
			ead::convert_to_node<double>(next_output_mask_));
		future_reward_ = age::add(ead::convert_to_node<double>(reward_),
			age::mul(
				ead::make_constant_scalar<double>(params_.discount_rate_,
					target_values->shape()),
				target_values)); // reward for each instance in batch

		// prediction error
		auto masked_output_score = age::reduce_sum_1d(
			age::mul(train_out_, ead::convert_to_node<double>(output_mask_)), 0);
		auto temp_diff = age::sub(masked_output_score, future_reward_);
		prediction_error_ = age::reduce_mean(age::pow(temp_diff,
			ead::make_constant_scalar<double>(2, temp_diff->shape())));

		// updates for source network
		pbm::PathedMapT svmap = source_qnet_->list_bases();
		std::unordered_map<std::string,size_t> labelled_indices;
		eqns::VariablesT source_vars;
		for (auto vpair : svmap)
		{
			if (auto var = std::dynamic_pointer_cast<
				ead::Variable<double>>(vpair.first))
			{
				auto label = fmts::to_string(
					vpair.second.begin(), vpair.second.end());
				labelled_indices.emplace(label, source_vars.size());
				source_vars.push_back(
					std::make_shared<ead::VariableNode<double>>(var));
			}
		}
		updates_ = update(prediction_error_, source_vars);

		// update target network
		pbm::PathedMapT tvmap = target_qnet_->list_bases();
		size_t nvars = source_vars.size();
		eqns::VariablesT target_vars(nvars);
		for (auto vpair : tvmap)
		{
			if (auto var = std::dynamic_pointer_cast<
				ead::Variable<double>>(vpair.first))
			{
				auto label = fmts::to_string(
					vpair.second.begin(), vpair.second.end());
				target_vars[labelled_indices[label]] =
					std::make_shared<ead::VariableNode<double>>(var);
			}
		}
		eqns::AssignsT target_assigns;
		for (size_t i = 0; i < nvars; i++)
		{
			// this is equivalent to target = (1-alpha) * target + alpha * source
			auto target = ead::convert_to_node<double>(target_vars[i]);
			auto source = ead::convert_to_node<double>(source_vars[i]);
			auto diff = age::sub(target, source);
			auto target_update_rate = ead::make_constant_scalar<double>(
				params_.target_update_rate_, diff->shape());

			auto target_next = age::sub(target, age::mul(
				target_update_rate, diff));
			target_assigns.push_back(eqns::VarAssign{target_vars[i], target_next});
		}
		updates_.push_back(target_assigns);

		std::vector<ead::NodeptrT<double>*> to_optimize =
		{
			&prediction_error_,
			&train_out_,
			&output_,
		};
		for (eqns::AssignsT& assigns : updates_)
		{
			for (eqns::VarAssign& assign : assigns)
			{
				to_optimize.push_back(&assign.source_);
			}
		}

		size_t n_roots = to_optimize.size();
		ead::NodesT<double> roots(n_roots);
		std::transform(to_optimize.begin(), to_optimize.end(),
			roots.begin(), [](ead::NodeptrT<double>* ptr) { return *ptr; });
		roots = ead::ops_reuse<double>(ead::multi_optimize<double>(roots));

		for (size_t i = 0; i < n_roots; ++i)
		{
			sess_->track(roots[i]);
			*to_optimize[i] = roots[i];
		}
	}

	uint8_t action (std::vector<double>& input)
	{
		actions_executed_++; // book keep
		double exploration = linear_annealing(1.0);
		// perform random exploration action
		if (get_random() < exploration)
		{
			return std::floor(get_random() * source_qnet_->get_noutput());
		}
		input_->assign(input.data(), input_->shape());
		sess_->update({input_->get_tensor().get()});
		ead::TensMapT<double>* doutput = output_->get_tensmap();
		double* dptr = doutput->data();
		uint8_t max_i = 0;
		for (uint8_t i = 1, n = ead::get_shape(*doutput).n_elems(); i < n; ++i)
		{
			if (dptr[max_i] < dptr[i])
			{
				max_i = i;
			}
		}
		return max_i;
	}

	void store (std::vector<double> observation, size_t action_idx,
		double reward, std::vector<double> new_obs)
	{
		if (0 == n_store_called_ % params_.store_interval_)
		{
			experiences_.push_back(ExpBatch{observation, action_idx, reward, new_obs});
			if (experiences_.size() > params_.max_exp_)
			{
				experiences_.front() = std::move(experiences_.back());
				experiences_.pop_back();
			}
		}
		n_store_called_++;
	}

	void train (void)
	{
		// extract mini_batch from buffer and backpropagate
		if (0 == (n_train_called_ % params_.train_interval_))
		{
			if (experiences_.size() < params_.mini_batch_size_) return;

			std::vector<ExpBatch> samples = random_sample();

			// batch data process
			std::vector<double> states; // <ninput, batchsize>
			std::vector<double> new_states; // <ninput, batchsize>
			std::vector<double> action_mask; // <noutput, batchsize>
			std::vector<double> new_states_mask; // <batchsize>
			std::vector<double> rewards; // <batchsize>

			for (size_t i = 0, n = samples.size(); i < n; i++)
			{
				ExpBatch batch = samples[i];
				states.insert(states.end(),
					batch.observation_.begin(), batch.observation_.end());
				{
					std::vector<double> local_act_mask(
						source_qnet_->get_noutput(), 0);
					local_act_mask[batch.action_idx_] = 1.0;
					action_mask.insert(action_mask.end(),
						local_act_mask.begin(), local_act_mask.end());
				}
				rewards.push_back(batch.reward_);
				if (batch.new_observation_.empty())
				{
					new_states.insert(new_states.end(),
						source_qnet_->get_ninput(), 0);
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

			sess_->update({
				train_input_->get_tensor().get(),
				output_mask_->get_tensor().get(),
				next_input_->get_tensor().get(),
				next_output_mask_->get_tensor().get(),
				reward_->get_tensor().get(),
			});
			assign_groups(*sess_, updates_);
			iteration_++;
		}
		n_train_called_++;
	}

	ead::NodeptrT<double> get_error (void) const
	{
		return prediction_error_;
	}

	size_t get_numtrained (void) const
	{
		return iteration_;
	}

	// === forward computation ===
	// fanin: shape <ninput>
	ead::VarptrT<double> input_ = nullptr;

	// fanout: shape <noutput>
	ead::NodeptrT<double> output_ = nullptr;

	// === backward computation ===
	// train fanin: shape <ninput, batchsize>
	ead::VarptrT<double> train_input_ = nullptr;

	// train fanout: shape <noutput, batchsize>
	ead::NodeptrT<double> train_out_ = nullptr;

	// === updates && optimizer ===
	eqns::AssignGroupsT updates_;

	ead::Session<double>* sess_;

private:
	// experience replay
	struct ExpBatch
	{
		std::vector<double> observation_;
		size_t action_idx_;
		double reward_;
		std::vector<double> new_observation_;
	};

	double linear_annealing (double initial_prob) const
	{
		if (actions_executed_ >= params_.exploration_period_)
			return params_.rand_action_prob_;
		return initial_prob - actions_executed_ * (initial_prob -
			params_.rand_action_prob_) / params_.exploration_period_;
	}

	double get_random (void)
	{
		return explore_(ead::get_engine());
	}

	std::vector<ExpBatch> random_sample (void)
	{
		std::vector<size_t> indices(experiences_.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::random_shuffle(indices.begin(), indices.end());
		std::vector<ExpBatch> res;
		for (size_t i = 0; i < params_.mini_batch_size_; i++)
		{
			size_t idx = indices[i];
			res.push_back(experiences_[idx]);
		}
		return res;
	}

	// === scalar parameters ===
	// training parameters
	DQNInfo params_;

	// source network
	modl::MLPptrT source_qnet_;

	// target network
	modl::MLPptrT target_qnet_;

	// === prediction computation ===
	// train_fanin: shape <ninput, batchsize>
	ead::VarptrT<double> next_input_ = nullptr;

	// train mask: shape <batchsize>
	ead::VarptrT<double> next_output_mask_ = nullptr;

	// train fanout: shape <noutput, batchsize>
	ead::NodeptrT<double> next_output_ = nullptr;

	// reward associated with next_output_: shape <batchsize>
	ead::VarptrT<double> reward_ = nullptr;

	// future reward calculated from reward history: <1, batchsize>
	ead::NodeptrT<double> future_reward_ = nullptr;

	// === q-value computation ===
	// weight output to get overall score: shape <noutput, batchsize>
	ead::VarptrT<double> output_mask_ = nullptr;

	// overall score: shape <noutput>
	ead::NodeptrT<double> score_ = nullptr;

	// future error that we want to minimize: scalar shape
	ead::NodeptrT<double> prediction_error_ = nullptr;

	// states
	size_t actions_executed_ = 0;

	size_t iteration_ = 0;

	size_t n_store_called_ = 0;

	size_t n_train_called_ = 0;

	std::uniform_real_distribution<double> explore_;

	std::vector<ExpBatch> experiences_;
};
