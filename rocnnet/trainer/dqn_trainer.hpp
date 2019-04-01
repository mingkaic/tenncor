#include "rocnnet/modl/mlp.hpp"

#include "ead/matcher/matcher.hpp"
struct DQNInfo
{
	DQNInfo (size_t train_interval = 5,
		PybindT rand_action_prob = 0.05,
		PybindT discount_rate = 0.95,
		PybindT target_update_rate = 0.01,
		PybindT exploration_period = 1000,
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
	PybindT rand_action_prob_ = 0.05;
	PybindT discount_rate_ = 0.95;
	PybindT target_update_rate_ = 0.01;
	PybindT exploration_period_ = 1000;
	// memory parameters
	size_t store_interval_ = 5;
	uint8_t mini_batch_size_ = 32;
	size_t max_exp_ = 30000;
};

struct DQNTrainer
{
	DQNTrainer (modl::MLPptrT brain, ead::Session<PybindT>& sess,
		eqns::ApproxFuncT update, DQNInfo param) :
		sess_(&sess),
		params_(param),
		source_qnet_(brain),
		target_qnet_(std::make_shared<modl::MLP>(*brain))
	{
		input_ = ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({source_qnet_->get_ninput()}), "observation");
		train_input_ = ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({source_qnet_->get_ninput(),
			params_.mini_batch_size_}), "train_observation");
		next_input_ = ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({source_qnet_->get_ninput(),
			params_.mini_batch_size_}), "next_observation");
		next_output_mask_ = ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({params_.mini_batch_size_}),
			"next_observation_mask");
		reward_ = ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({params_.mini_batch_size_}), "rewards");
		output_mask_ = ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({source_qnet_->get_noutput(),
			params_.mini_batch_size_}), "action_mask");

		// forward action score computation
		output_ = (*source_qnet_)(ead::convert_to_node<PybindT>(input_));

		train_out_ = (*source_qnet_)(ead::convert_to_node<PybindT>(train_input_));

		// predicting target future rewards
		next_output_ = (*target_qnet_)(ead::convert_to_node<PybindT>(next_input_));

		auto target_values = age::mul(
			age::reduce_max_1d(next_output_, 0),
			ead::convert_to_node<PybindT>(next_output_mask_));
		future_reward_ = age::add(ead::convert_to_node<PybindT>(reward_),
			age::mul(
				ead::make_constant_scalar<PybindT>(params_.discount_rate_,
					target_values->shape()),
				target_values)); // reward for each instance in batch

		// prediction error
		auto masked_output_score = age::reduce_sum_1d(
			age::mul(train_out_, ead::convert_to_node<PybindT>(output_mask_)), 0);
		auto temp_diff = age::sub(masked_output_score, future_reward_);
		prediction_error_ = age::reduce_mean(age::square(temp_diff));

		// updates for source network
		pbm::PathedMapT svmap = source_qnet_->list_bases();
		std::unordered_map<std::string,size_t> labelled_indices;
		eqns::VariablesT source_vars;
		for (auto vpair : svmap)
		{
			if (auto var = std::dynamic_pointer_cast<
				ead::Variable<PybindT>>(vpair.first))
			{
				auto label = fmts::join("::",
					vpair.second.begin(), vpair.second.end());
				labelled_indices.emplace(label, source_vars.size());

				auto vnode = std::make_shared<ead::VariableNode<PybindT>>(var);
				vnode->set_label(label);
				source_vars.push_back(vnode);
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
				ead::Variable<PybindT>>(vpair.first))
			{
				auto label = fmts::to_string(
					vpair.second.begin(), vpair.second.end());
				target_vars[labelled_indices[label]] =
					std::make_shared<ead::VariableNode<PybindT>>(var);
			}
		}
		eqns::AssignsT target_assigns;
		for (size_t i = 0; i < nvars; i++)
		{
			// this is equivalent to target = (1-alpha) * target + alpha * source
			auto target = ead::convert_to_node<PybindT>(target_vars[i]);
			auto source = ead::convert_to_node<PybindT>(source_vars[i]);
			auto diff = age::sub(target, source);
			auto target_update_rate = ead::make_constant_scalar<PybindT>(
				params_.target_update_rate_, diff->shape());

			auto target_next = age::sub(target, age::mul(
				target_update_rate, diff));
			target_assigns.push_back(eqns::VarAssign{
				fmts::sprintf("::target_grad_%s",
					target_vars[i]->get_label().c_str()),
				target_vars[i], target_next});
		}
		updates_.push_back(target_assigns);

		std::vector<ead::NodeptrT<PybindT>*> to_optimize =
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

		opt::GraphOpt optimizer = opt::config_opt();
		size_t n_roots = to_optimize.size();
		ead::NodesT<PybindT> roots(n_roots);
		std::transform(to_optimize.begin(), to_optimize.end(), roots.begin(),
			[&optimizer](ead::NodeptrT<PybindT>* ptr)
			{
				(*ptr)->get_tensor()->accept(optimizer);
				return *ptr;
			});

		{
			auto temp = optimizer.apply_optimization(roots);
			roots = ead::ops_reuse<PybindT>(ead::multi_optimize<PybindT>(temp));
		}

		for (size_t i = 0; i < n_roots; ++i)
		{
			sess_->track(roots[i]);
			*to_optimize[i] = roots[i];
		}
	}

	uint8_t action (std::vector<PybindT>& input)
	{
		actions_executed_++; // book keep
		PybindT exploration = linear_annealing(1.0);
		// perform random exploration action
		if (get_random() < exploration)
		{
			return std::floor(get_random() * source_qnet_->get_noutput());
		}
		input_->assign(input.data(), input_->shape());
		sess_->update({input_->get_tensor().get()});
		PybindT* dptr = output_->data();
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

	void store (std::vector<PybindT> observation, size_t action_idx,
		PybindT reward, std::vector<PybindT> new_obs)
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
			std::vector<PybindT> states; // <ninput, batchsize>
			std::vector<PybindT> new_states; // <ninput, batchsize>
			std::vector<PybindT> action_mask; // <noutput, batchsize>
			std::vector<PybindT> new_states_mask; // <batchsize>
			std::vector<PybindT> rewards; // <batchsize>

			for (size_t i = 0, n = samples.size(); i < n; i++)
			{
				ExpBatch batch = samples[i];
				states.insert(states.end(),
					batch.observation_.begin(), batch.observation_.end());
				{
					std::vector<PybindT> local_act_mask(
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
			assign_groups(updates_,
				[this](std::unordered_set<ade::iTensor*>& updated)
				{
					this->sess_->update(updated);
				});
			iteration_++;
		}
		n_train_called_++;
	}

	ead::NodeptrT<PybindT> get_error (void) const
	{
		return prediction_error_;
	}

	size_t get_numtrained (void) const
	{
		return iteration_;
	}

	// === forward computation ===
	// fanin: shape <ninput>
	ead::VarptrT<PybindT> input_ = nullptr;

	// fanout: shape <noutput>
	ead::NodeptrT<PybindT> output_ = nullptr;

	// === backward computation ===
	// train fanin: shape <ninput, batchsize>
	ead::VarptrT<PybindT> train_input_ = nullptr;

	// train fanout: shape <noutput, batchsize>
	ead::NodeptrT<PybindT> train_out_ = nullptr;

	// === updates && optimizer ===
	eqns::AssignGroupsT updates_;

	ead::Session<PybindT>* sess_;

private:
	// experience replay
	struct ExpBatch
	{
		std::vector<PybindT> observation_;
		size_t action_idx_;
		PybindT reward_;
		std::vector<PybindT> new_observation_;
	};

	PybindT linear_annealing (PybindT initial_prob) const
	{
		if (actions_executed_ >= params_.exploration_period_)
			return params_.rand_action_prob_;
		return initial_prob - actions_executed_ * (initial_prob -
			params_.rand_action_prob_) / params_.exploration_period_;
	}

	PybindT get_random (void)
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
	ead::VarptrT<PybindT> next_input_ = nullptr;

	// train mask: shape <batchsize>
	ead::VarptrT<PybindT> next_output_mask_ = nullptr;

	// train fanout: shape <noutput, batchsize>
	ead::NodeptrT<PybindT> next_output_ = nullptr;

	// reward associated with next_output_: shape <batchsize>
	ead::VarptrT<PybindT> reward_ = nullptr;

	// future reward calculated from reward history: <1, batchsize>
	ead::NodeptrT<PybindT> future_reward_ = nullptr;

	// === q-value computation ===
	// weight output to get overall score: shape <noutput, batchsize>
	ead::VarptrT<PybindT> output_mask_ = nullptr;

	// overall score: shape <noutput>
	ead::NodeptrT<PybindT> score_ = nullptr;

	// future error that we want to minimize: scalar shape
	ead::NodeptrT<PybindT> prediction_error_ = nullptr;

	// states
	size_t actions_executed_ = 0;

	size_t iteration_ = 0;

	size_t n_store_called_ = 0;

	size_t n_train_called_ = 0;

	std::uniform_real_distribution<PybindT> explore_;

	std::vector<ExpBatch> experiences_;
};
