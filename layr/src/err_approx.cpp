#include "eteq/generated/api.hpp"

#include "layr/err_approx.hpp"

#ifdef LAYR_ERR_APPROX_HPP

namespace layr
{

AssignGroupsT sgd (const VarErrsT& leaves,
	PybindT learning_rate, std::string root_label)
{
	AssignsT assignments;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = eteq::convert_to_node(leaves[i].first);
		auto err = leaves[i].second;
		auto next = leaf_node - err * learning_rate;
		assignments.push_back(VarAssign{
			fmts::sprintf("sgd::%s_grad_%s",
				root_label.c_str(), leaves[i].first->to_string().c_str()),
			leaves[i].first, next});
	}
	return {assignments};
}

AssignGroupsT rms_momentum (const VarErrsT& leaves, PybindT learning_rate,
	PybindT discount_factor, PybindT epsilon, std::string root_label)
{
	// assign momentums before leaves
	AssignsT momentum_assigns;
	AssignsT leaf_assigns;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = eteq::convert_to_node(leaves[i].first);
		auto err = leaves[i].second;
		teq::Shape eshape = err->shape();
		eteq::VarptrT<PybindT> momentum =
			eteq::make_variable_scalar<PybindT>(1, eshape, "momentum");
		auto momentum_node = eteq::convert_to_node(momentum);

		auto momentum_next = discount_factor * momentum_node +
			PybindT(1. - discount_factor) * tenncor::square(err);
		auto leaf_next = leaf_node - err * learning_rate /
			(tenncor::sqrt(momentum_node) + epsilon);
		momentum_assigns.push_back(VarAssign{
			fmts::sprintf("rms_momentum::%s_momentum_%s",
				root_label.c_str(), leaves[i].first->to_string().c_str()),
			momentum, momentum_next});
		leaf_assigns.push_back(VarAssign{
			fmts::sprintf("rms_momentum::%s_grad_%s",
				root_label.c_str(), leaves[i].first->to_string().c_str()),
			leaves[i].first, leaf_next});
	}
	return {momentum_assigns, leaf_assigns};
}

void assign_groups (const AssignGroupsT& groups, UpdateStepF update_step)
{
	for (const AssignsT& group : groups)
	{
		teq::TensSetT updated_var;
		for (const layr::VarAssign& assign : group)
		{
			updated_var.emplace(assign.target_->get_tensor().get());
			assign.target_->assign(assign.source_->data(),
				assign.source_->shape());
		}
		update_step(updated_var);
	}
}

void assign_groups_preupdate (const AssignGroupsT& groups, UpdateStepF update_step)
{
	for (const AssignsT& group : groups)
	{
		teq::TensSetT sources;
		for (const layr::VarAssign& assign : group)
		{
			sources.emplace(assign.source_->get_tensor().get());
		}
		update_step(sources);
		for (const layr::VarAssign& assign : group)
		{
			assign.target_->assign(assign.source_->data(),
				assign.source_->shape());
		}
	}
}

}

#endif
