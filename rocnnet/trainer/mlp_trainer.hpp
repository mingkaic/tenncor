#include "eteq/grader.hpp"

#include "layr/seqmodel.hpp"
#include "layr/err_approx.hpp"

#ifndef RCN_MLP_TRAINER_HPP
#define RCN_MLP_TRAINER_HPP

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
		train_in_(eteq::make_variable_scalar<PybindT>(0., teq::Shape({
			(teq::DimT) model.get_ninput(), batch_size}), "train_in")),
		expected_out_(eteq::make_variable_scalar<PybindT>(0., teq::Shape({
			(teq::DimT) model.get_noutput(), batch_size}), "expected_out")),
		sess_(&sess),
		ctx_(ctx)
	{
		auto train_out = model.connect(
			eteq::convert_to_node<PybindT>(train_in_));
		error_ = tenncor::square(
			eteq::convert_to_node<PybindT>(expected_out_) - train_out);

		auto contents = model.get_contents();
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
			train_out->get_tensor(),
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

	void train (
		const eteq::ShapedArr<PybindT>& train_in,
		const eteq::ShapedArr<PybindT>& expected_out)
	{
		train_in_->assign(train_in);
		expected_out_->assign(expected_out);

		sess_->update();
		assign_groups(updates_,
			[this](std::unordered_set<teq::iTensor*>& updated)
			{
				this->sess_->update();
			});
		++ctx_.n_iterations_;
	}

	eteq::VarptrT<PybindT> train_in_;
	eteq::VarptrT<PybindT> expected_out_;
	NodeptrT error_;

	layr::AssignGroupsT updates_;
	eteq::iSession* sess_;

	TrainingContext ctx_;
};

}

#endif // RCN_MLP_TRAINER_HPP
