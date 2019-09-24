#include "eteq/grader.hpp"

#include "layr/model.hpp"

#include "rocnnet/layr/err_approx.hpp"

#ifndef LAYR_MLP_TRAINER_HPP
#define LAYR_MLP_TRAINER_HPP

namespace trainer
{

// Normal default context that only stores the number of iterations
struct TrainingContext final
{
	size_t n_iterations_ = 0;
};

// MLPTrainer does not own anything
struct MLPTrainer final
{
	MLPTrainer (layr::SequentialModel& model,
		eteq::iSession& sess, layr::ApproxF update, teq::DimT batch_size,
		layr::NodeUnarF gradprocess = layr::NodeUnarF(layr::identity),
		TrainingContext ctx = TrainingContext()) :
		batch_size_(batch_size),
		train_in_(eteq::make_variable_scalar<PybindT>(0.0, teq::Shape({
			(teq::DimT) model.get_ninput(), batch_size}), "train_in")),
		model_(model),
		sess_(&sess),
		ctx_(ctx)
	{
		train_out_ = model_.connect(
			eteq::convert_to_node<PybindT>(train_in_));
		expected_out_ = eteq::make_variable_scalar<PybindT>(0.0, teq::Shape({
			(teq::DimT) model.get_noutput(), batch_size}), "expected_out");
		error_ = tenncor::square(
			eteq::convert_to_node<PybindT>(expected_out_) - train_out_);

		auto contents = model_.get_contents();
		layr::VarErrsT vars;
		for (auto tens : contents)
		{
			if (auto var = std::dynamic_pointer_cast<
				eteq::Variable<PybindT>>(tens))
			{
				auto varnode = std::make_shared<eteq::VariableNode<PybindT>>(var);
				vars.push_back({
					varnode,
					gradprocess(eteq::derive(error_, eteq::convert_to_node(varnode)))
				});
			}
		}
		updates_ = update(vars);

		teq::TensT track_batch = {
			train_out_->get_tensor(),
			error_->get_tensor(),
		};
		for (layr::AssignsT& assigns : updates_)
		{
			for (layr::VarAssign& assign : assigns)
			{
				track_batch.push_back(assign.source_->get_tensor());
			}
		}
		sess_->track(track_batch);
	}

	void train (std::vector<PybindT>& train_in,
		std::vector<PybindT>& expected_out)
	{
		size_t insize = model_.get_ninput();
		size_t outsize = model_.get_noutput();
		if (train_in.size() != insize * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match "
				"input size (%d) * batchsize (%d)", train_in.size(),
				insize, batch_size_);
		}
		if (expected_out.size() != outsize * batch_size_)
		{
			logs::fatalf("expected output size (%d) does not match "
				"output size (%d) * batchsize (%d)", expected_out.size(),
				outsize, batch_size_);
		}
		train_in_->assign(train_in.data(), train_in_->shape());
		expected_out_->assign(expected_out.data(), expected_out_->shape());

		sess_->update({
			train_in_->get_tensor().get(),
			expected_out_->get_tensor().get(),
		});
		assign_groups(updates_,
			[this](std::unordered_set<teq::iTensor*>& updated)
			{
				this->sess_->update(updated);
			});
		++ctx_.n_iterations_;
	}

	layr::SequentialModel& model_;

	uint8_t batch_size_;
	eteq::VarptrT<PybindT> train_in_;
	eteq::VarptrT<PybindT> expected_out_;
	eteq::NodeptrT<PybindT> train_out_;
	eteq::NodeptrT<PybindT> error_;

	layr::AssignGroupsT updates_;
	eteq::iSession* sess_;

	TrainingContext ctx_;
};

}

#endif // LAYR_MLP_TRAINER_HPP
