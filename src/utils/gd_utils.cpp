//
// Created by Mingkai Chen on 2017-04-27.
//

#include "include/utils/gd_utils.hpp"

#ifdef TENNCOR_GD_UTILS_HPP

namespace nnet
{
	
gd_updater::gd_updater (double learning_rate) : learning_rate_(learning_rate) {}

gd_updater::~gd_updater(void) {}

gd_updater* gd_updater::clone (void) const { return clone_impl(); }

gd_updater* gd_updater::move (void) { return move_impl(); }

updates_t gd_updater::calculate (inode* root, grad_process intermediate_process)
{
	std::vector<updater_t > updates;
	std::unordered_set<ileaf*> leafset = root->get_leaves();
	std::vector<std::pair<inode*,variable*> > gress;
	for (ileaf* l : leafset)
	{
		variable* Wb = dynamic_cast<variable*>(l);
		if (Wb && ignored_.end() == ignored_.find(Wb))
		{
			gress.push_back({root->derive(Wb), Wb});
		}
	}

	for (auto& gpair : gress)
	{
		varptr gres = gpair.first;
		updates.push_back(process_update(gres, gpair.second, intermediate_process));
	}
	return updates;
}

void gd_updater::ignore_subtree (inode* subroot)
{
	std::unordered_set<ileaf*> leafset = subroot->get_leaves();
	for (ileaf* l : leafset)
	{
		if (variable* Wb = dynamic_cast<variable*>(l))
		{
			ignored_.emplace(Wb);
		}
	}
}

void gd_updater::clear_ignore (void)
{
	ignored_.clear();
}

void gd_updater::set_learning_rate (double learning_rate)
{
	learning_rate_ = learning_rate;
}


vgb_updater::vgb_updater (double learning_rate) : gd_updater(learning_rate) {}
		
vgb_updater*  vgb_updater::clone (void) { return static_cast<vgb_updater*>(clone_impl()); }

vgb_updater*  vgb_updater::move (void) { return static_cast<vgb_updater*>(move_impl()); }

gd_updater*  vgb_updater::clone_impl (void) const
{
	return new vgb_updater(*this);
}

gd_updater* vgb_updater::move_impl (void)
{
	return new vgb_updater(std::move(*this));
}

updater_t vgb_updater::process_update (varptr& gres,
	variable* leaf, grad_process intermediate_process)
{
	// leaf = leaf - learning_rate * gres
	varptr leaf_step = intermediate_process(gres, leaf) * learning_rate_;
	return [leaf, leaf_step](bool notify)
	{
		leaf->assign_sub(, notify);
	}
}


momentum_updater* momentum_updater::clone (void) { return static_cast<momentum_updater*>(clone_impl()); }

momentum_updater* momentum_updater::move (void) { return static_cast<momentum_updater*>(move_impl()); }

gd_updater* momentum_updater::clone_impl (void) const
{
	return new momentum_updater(*this);
}

gd_updater* momentum_updater::move_impl (void)
{
	return new momentum_updater(std::move(*this));
}

updater_t momentum_updater::process_update (varptr& /*gres*/,
	variable* /*leaf*/, grad_process /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](bool) {};
}


adadelta_updater* adadelta_updater::clone (void) { return static_cast<adadelta_updater*>(clone_impl()); }

adadelta_updater* adadelta_updater::move (void) { return static_cast<adadelta_updater*>(move_impl()); }

gd_updater* adadelta_updater::clone_impl (void) const
{
	return new adadelta_updater(*this);
}

gd_updater* adadelta_updater::move_impl (void)
{
	return new adadelta_updater(std::move(*this));
}

updater_t adadelta_updater::process_update (varptr& /*gres*/,
	variable* /*leaf*/, grad_process /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](bool) {};
}


adagradupdater* adagradupdater::clone (void) { return static_cast<adagradupdater*>(clone_impl()); }

adagradupdater* adagradupdater::move (void) { return static_cast<adagradupdater*>(move_impl()); }

gd_updater* adagradupdater::clone_impl (void) const
{
	return new adagradupdater(*this);
}

gd_updater* adagradupdater::move_impl (void)
{
	return new adagradupdater(std::move(*this));
}

updater_t adagradupdater::process_update (varptr& /*gres*/,
	variable* /*leaf*/, grad_process /*intermediate_process*/)
{
	throw std::bad_function_call();
	return [](bool) {};
}


rmspropupdater::rmspropupdater (double learning_rate, double discount_factor) :
	gd_updater(learning_rate),
	discount_factor_(discount_factor) {}

rmspropupdater::~rmspropupdater (void)
{
	for (variable* momentum : momentums_)
	{
		delete momentum;
	}
}

rmspropupdater* rmspropupdater::clone (void) { return static_cast<rmspropupdater*>(clone_impl()); }

rmspropupdater* rmspropupdater::move (void) { return static_cast<rmspropupdater*>(move_impl()); }

gd_updater* rmspropupdater::clone_impl (void) const
{
	return new rmspropupdater(*this);
}

gd_updater* rmspropupdater::move_impl (void)
{
	return new rmspropupdater(std::move(*this));
}

rmspropupdater& rmspropupdater::operator = (const rmspropupdater& other)
{
	if (this != &other)
	{
		gd_updater::operator = (other);
		discount_factor_ = other.discount_factor_;
	}
	return *this;
}

rmspropupdater& rmspropupdater::operator = (rmspropupdater&& other)
{
	if (this != &other)
	{
		gd_updater::operator = (std::move(other));
		discount_factor_ = std::move(other.discount_factor_);
	}
	return *this;
}

void rmspropupdater::set_discount_factor (double discount_factor)
{
	discount_factor_ = discount_factor;
}

rmspropupdater::rmspropupdater (const rmspropupdater& other) :
	gd_updater(other),
	discount_factor_(other.discount_factor_) {}

rmspropupdater::rmspropupdater (rmspropupdater&& other) :
	gd_updater(std::move(other)),
	discount_factor_(std::move(other.discount_factor_)) {}
	
updater_t rmspropupdater::process_update (varptr& gres,
	variable* leaf, grad_process intermediate_process)
{
	const_init wuninit((double) 1);
	variable* momentum = new variable(leaf->get_shape(), wuninit, DOUBLE, "momentum");
	momentum->initialize();
	momentums_.push_back(momentum); // updater manages momentum variable

	// momentum = discount_factor_ * momentum + (1 - discount_factor_) * gres^2
	// leaf = leaf - learning_rate * gres / (sqrt(momentum) + epsilon)
	varptr dres = intermediate_process(gres, leaf);
	varptr momentum_step = discount_factor_ * varptr(momentum) + (1-discount_factor_) * pow(dres, 2);
	varptr leaf_step = dres * learning_rate_ / (sqrt(momentum_step) + epsilon_);
	return [momentum, momentum_step, leaf, leaf_step](bool notify)
	{
		momentum->assign(momentum_step, false);
		leaf->assign_sub(leaf_step, notify);
	};
}


}

#endif