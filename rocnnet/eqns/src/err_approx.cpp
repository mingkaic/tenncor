#include "ead/generated/api.hpp"

#include "ead/grader.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifdef EQNS_ERR_APPROX_HPP

namespace eqns
{

ead::NodeptrT<PybindT> identity (ead::NodeptrT<PybindT> node)
{
	return node;
}

AssignGroupsT sgd (ead::NodeptrT<PybindT>& root, VariablesT leaves,
	PybindT learning_rate, NodeUnarF gradprocess, std::string root_label)
{
	AssignsT assignments;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = ead::convert_to_node(leaves[i]);
		auto grad = ead::derive<PybindT>(root, leaf_node);
		// given root = f, err(x) ~ x - η * df(x), where η is the learning rate
		ade::Shape gshape = grad->shape();
		auto next = age::sub(leaf_node,
			age::mul(gradprocess(grad),
				ead::make_constant_scalar<PybindT>(learning_rate, gshape)));
		assignments.push_back(VarAssign{
			fmts::sprintf("sgd::%s_grad_%s",
				root_label.c_str(), leaves[i]->get_label().c_str()),
			leaves[i], next});
	}
	return {assignments};
}

AssignGroupsT rms_momentum (ead::NodeptrT<PybindT>& root, VariablesT leaves,
	PybindT learning_rate, PybindT discount_factor, PybindT epsilon,
	NodeUnarF gradprocess, std::string root_label)
{
	// assign momentums before leaves
	AssignsT momentum_assigns;
	AssignsT leaf_assigns;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = ead::convert_to_node(leaves[i]);
		auto grad = ead::derive<PybindT>(root, leaf_node);
		// next_x ~ x - (η * df(x)) / (sqrt(ε + momentum_next))
		// momentum_next ~ χ * momentum_prev + (1 - χ) * df(x) ^ 2
		//
		// where root = f, η is the learning rate, ε is epsilon,
		// and χ is discount_factor
		ade::Shape gshape = grad->shape();
		ead::VarptrT<PybindT> momentum =
			ead::make_variable_scalar<PybindT>(1, gshape, "momentum");
		auto momentum_node = ead::convert_to_node(momentum);
		ead::NodeptrT<PybindT> discount_node =
			ead::make_constant_scalar<PybindT>(discount_factor, gshape);
		ead::NodeptrT<PybindT> datcount_node =
			ead::make_constant_scalar<PybindT>(1.0 - discount_factor, gshape);

		auto momentum_next = age::add(
				age::mul(discount_node, momentum_node),
				age::mul(datcount_node, age::square(grad))
			);
		auto leaf_next = age::sub(leaf_node,
			age::div(
				age::mul(gradprocess(grad),
					ead::make_constant_scalar<PybindT>(learning_rate, gshape)),
				age::add(age::sqrt(momentum_node),
					ead::make_constant_scalar<PybindT>(epsilon, gshape))
			));
		momentum_assigns.push_back(VarAssign{
			fmts::sprintf("rms_momentum::%s_momentum_%s",
				root_label.c_str(), leaves[i]->get_label().c_str()),
			momentum, momentum_next});
		leaf_assigns.push_back(VarAssign{
			fmts::sprintf("rms_momentum::%s_grad_%s",
				root_label.c_str(), leaves[i]->get_label().c_str()),
			leaves[i], leaf_next});
	}
	return {momentum_assigns, leaf_assigns};
}

void assign_groups (AssignGroupsT& groups, UpdateStepT update_step)
{
	for (AssignsT& group : groups)
	{
		std::unordered_set<ade::iTensor*> updated_var;
		for (eqns::VarAssign& assign : group)
		{
			updated_var.emplace(assign.target_->get_tensor().get());
			assign.target_->assign(assign.source_->data(),
				assign.source_->shape());
		}
		update_step(updated_var);
	}
}

}

#endif
