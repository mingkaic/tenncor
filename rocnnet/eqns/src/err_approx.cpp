#include "ead/generated/api.hpp"

#include "ead/grader.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifdef EQNS_ERR_APPROX_HPP

namespace eqns
{

AssignGroupsT sgd (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate)
{
	AssignsT assignments;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = ead::convert_to_node(leaves[i]);
		auto grad = ead::derive<double>(root, leaf_node);
		// given root = f, err(x) ~ x - η * df(x), where η is the learning rate
		ade::Shape gshape = grad->shape();
		auto next = age::sub(leaf_node,
			age::mul(grad, ead::make_constant_scalar<double>(learning_rate, gshape)));
		assignments.push_back(VarAssign{leaves[i], next});
	}
	return {assignments};
}

AssignGroupsT rms_momentum (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate, double discount_factor, double epsilon)
{
	// assign momentums before leaves
	AssignsT momentum_assigns;
	AssignsT leaf_assigns;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = ead::convert_to_node(leaves[i]);
		auto grad = ead::derive<double>(root, leaf_node);
		// next_x ~ x - (η * df(x)) / (sqrt(ε + momentum_next))
		// momentum_next ~ χ * momentum_prev + (1 - χ) * df(x) ^ 2
		//
		// where root = f, η is the learning rate, ε is epsilon,
		// and χ is discount_factor
		ade::Shape gshape = grad->shape();
		ead::VarptrT<double> momentum =
			ead::make_variable_scalar<double>(1, gshape, "momentum");
		auto momentum_node = ead::convert_to_node(momentum);
		ead::NodeptrT<double> discount_node =
			ead::make_constant_scalar<double>(discount_factor, gshape);
		ead::NodeptrT<double> datcount_node =
			ead::make_constant_scalar<double>(1.0 - discount_factor, gshape);

		auto momentum_next = age::add(
				age::mul(discount_node, momentum_node),
				age::mul(datcount_node, age::square(grad))
			);
		auto leaf_next = age::sub(leaf_node,
			age::div(
				age::mul(grad, ead::make_constant_scalar<double>(learning_rate, gshape)),
				age::add(age::sqrt(momentum_node),
					ead::make_constant_scalar<double>(epsilon, gshape))
			));
		momentum_assigns.push_back(VarAssign{momentum, momentum_next});
		leaf_assigns.push_back(VarAssign{leaves[i], leaf_next});
	}
	return {momentum_assigns, leaf_assigns};
}

void assign_groups (ead::Session<double>& sess, AssignGroupsT& groups)
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
		sess.update(updated_var);
	}
}

}

#endif
