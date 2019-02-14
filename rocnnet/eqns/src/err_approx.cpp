#include "ead/generated/api.hpp"

#include "ead/grader.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifdef EQNS_ERR_APPROX_HPP

namespace eqns
{

void assign_all (ead::Session<double>& sess, VarmapT connection)
{
	std::unordered_set<ade::iTensor*> updates;
	for (auto cpair : connection)
	{
		updates.emplace(cpair.first->get_tensor().get());
		auto tmap = cpair.second->get_tensmap();
		cpair.first->assign(tmap->data(),
			cpair.second->shape());
	}
	sess.update(updates);
}

Deltas sgd (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate)
{
	Deltas errs;
	VarmapT connection;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = ead::convert_to_node(leaves[i]);
		auto grad = ead::derive<double>(root, leaf_node);
		// given root = f, err(x) ~ x - η * df(x), where η is the learning rate
		ade::Shape gshape = grad->shape();
		auto next = age::sub(leaf_node,
			age::mul(grad, ead::make_constant_scalar<double>(learning_rate, gshape)));
		errs.upkeep_.push_back(next);
		connection.emplace(leaves[i], next);
	}
	errs.actions_.push_back(
		[connection](ead::Session<double>& sess)
		{
			assign_all(sess, connection);
		});
	return errs;
}

Deltas rms_momentum (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate, double discount_factor, double epsilon)
{
	Deltas errs;
	VarmapT momentum_connection;
	VarmapT leaf_connection;
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
		errs.upkeep_.push_back(momentum_node);
		errs.upkeep_.push_back(momentum_next);
		errs.upkeep_.push_back(leaf_next);
		momentum_connection.emplace(momentum, momentum_next);
		leaf_connection.emplace(leaves[i], leaf_next);
	}
	errs.actions_.push_back(
		[momentum_connection](ead::Session<double>& sess)
		{
			assign_all(sess, momentum_connection);
		});
	errs.actions_.push_back(
		[leaf_connection](ead::Session<double>& sess)
		{
			assign_all(sess, leaf_connection);
		});
	return errs;
}

}

#endif
