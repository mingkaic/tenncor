//
//  iconnector.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <queue>

#include "include/graph/connector/iconnector.hpp"

#ifdef TENNCOR_ICONNECTOR_HPP

namespace nnet
{

inline std::vector<iconnector*> to_con (std::vector<inode*> args)
{
	std::vector<iconnector*> conns;
	for (inode* a : args)
	{
		if (iconnector* con = dynamic_cast<iconnector*>(a))
		{
			conns.push_back(con);
		}
	}
	return conns;
}

iconnector::~iconnector (void)
{
	if (g_man_) g_man_->suicide(this);
}

iconnector* iconnector::clone (void) const
{
	return static_cast<iconnector*>(this->clone_impl());
}

iconnector* iconnector::move (void)
{
	return static_cast<iconnector*>(this->move_impl());
}

iconnector& iconnector::operator = (const iconnector& other)
{
	if (this != &other)
	{
		iobserver::operator = (other);
		inode::operator = (other);
		copy_helper(other);
	}
	return *this;
}

iconnector& iconnector::operator = (iconnector&& other)
{
	if (this != &other)
	{
		iobserver::operator = (std::move(other));
		inode::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

std::string iconnector::get_name (void) const
{
	std::string args = "";
	auto it = this->dependencies_.begin();
	auto et = this->dependencies_.end();
	const inode * arg = dynamic_cast<const inode*>(*it);
	while (args.empty() && nullptr == arg)
	{
		arg = dynamic_cast<const inode*>(*++it);
	}
	if (arg)
	{
		args = arg->get_label();
		++it;
	}
	while (it != et)
	{
		if (nullptr != (arg = dynamic_cast<const inode*>(*it)))
		{
			args += "," + arg->get_label();
		}
		it++;
	}
	return inode::get_name() + "(" + args + ")";
}

size_t iconnector::get_depth (void) const
{
	return depth_;
}

std::vector<inode*> iconnector::get_arguments (void) const
{
	std::vector<inode*> node_args(this->dependencies_.size());
	std::transform(this->dependencies_.begin(), this->dependencies_.end(), node_args.begin(),
		[](subject* s) { return static_cast<inode*>(s); });
	return node_args;
}

size_t iconnector::n_arguments (void) const
{
	return this->dependencies_.size();
}

const itensor* iconnector::eval (void)
{
	if (this->g_man_ && false == this->g_man_->freeze_)
	{
		this->g_man_->update();
	}
	return this->get_eval();
}

bool iconnector::is_same_graph (const iconnector* other) const
{
	return g_man_ == other->g_man_;
}

bool iconnector::potential_descendent (const iconnector* n) const
{
	// A is a descendent of B iff A's leaf set is a subset of B's leaf set (or vice versa)
	std::unordered_set<ileaf*> mine = this->get_leaves();
	std::unordered_set<ileaf*> their = n->get_leaves();

	if (mine.size() < their.size()) return false;
	for (ileaf* t : their)
	{
		if (mine.end() == mine.find(t))
		{
			return false;
		}
	}
	return true;
}

void iconnector::set_jacobian_front (JTRANSFER jac, std::vector<variable*> leaves)
{
	for (variable* l : leaves)
	{
		jacobians_[l].list_.push_front({jac, this});
	}
}

void iconnector::set_jacobian_back (JTRANSFER jac, std::vector<variable*> leaves)
{
	for (variable* l : leaves)
	{
		jacobians_[l].list_.push_back({jac, this});
	}
}

void iconnector::freeze_status (bool freeze)
{
	assert(this->g_man_);
	if (freeze)
	{
		this->g_man_->update();
	}
	this->g_man_->freeze_ = freeze;
}

iconnector::iconnector (std::vector<inode*> dependencies, std::string label) :
	inode(label),
	iobserver(std::vector<subject*>(dependencies.begin(), dependencies.end()))
{
	size_t n = dependencies.size();
	if (n> 0)
	{
		std::vector<size_t> depths(n, 0);
		std::transform(dependencies.begin(), dependencies.end(), depths.begin(),
		[](inode* n)
		{
			return n->get_depth();
		});
		depth_ = *(std::max_element(depths.begin(), depths.end())) + 1;
	}

	std::unordered_set<inode*> deps;
	// todo: test for jacobian, and leaf transfer
	// if we have more than 1 jacobian, separate the operators for each branch
	for (subject* sub : this->dependencies_)
	{
		inode* arg = static_cast<inode*>(sub);
		// only perform following on unique dependent nodes:
		if (deps.end() == deps.find(arg))
		{
			if (iconnector* imm = dynamic_cast<iconnector*>(arg))
			{
				for (auto jpair : imm->jacobians_)
				{
					variable* leaf = jpair.first;
					auto j = jpair.second;
					if (false == j.list_.empty())
					{
						auto jit = this->jacobians_.find(leaf);
						if (this->jacobians_.end() == jit)
						{
							this->jacobians_[leaf] = j;
							this->jacobians_[leaf].terminal_ = false;
						}
						else if (j.uid_ != jit->second.uid_)
						{
							this->jacobians_[leaf].terminal_ = true; // terminate
							this->jacobians_[leaf].list_.clear();
						}
					}
				}
			}
			deps.emplace(arg);
		}
	}
	update_graph(to_con(dependencies));
}

iconnector::iconnector (const iconnector& other) :
	inode(other),
	iobserver(other)
{
	copy_helper(other);
}

iconnector::iconnector (iconnector&& other) :
	inode(std::move(other)),
	iobserver(std::move(other))
{
	move_helper(std::move(other));
}

void iconnector::update_graph (std::vector<iconnector*> args)
{
	if (args.empty())
	{
		if (nullptr == g_man_)
		{
			graph_manager::get(this);
		}
		return;
	}
	g_man_ = graph_manager::get(args[0], this);
	for (size_t i = 1, n = args.size(); i < n; i++)
	{
		g_man_->consume(args[i]->g_man_);
	}
}

varptr iconnector::jacobian_call (varptr out, variable* leaf) const
{
	auto jpair = this->jacobians_.find(leaf);
	if (this->jacobians_.end() != jpair)
	{
		auto& jlist = jpair->second.list_;
		for (auto it = jlist.rbegin(), et = jlist.rend(); it != et; it++)
		{
			const JTRANSFER& jt = it->first;
			// get the node where jacobian originate from
			const inode* orig = it->second;
			// get origin arguments and its gradients
			std::vector<inode*> args = orig->get_arguments();
			std::vector<inode*> grads(args.size(), nullptr);
			std::transform(args.begin(), args.end(), grads.begin(),
			[this, leaf](inode* arg)
			{
				return this->take_gradient(arg, leaf);
			});
			// operate on out using args and grad
			out = jt(out, args, grads);
		}
	}
	return out;
}

void iconnector::copy_helper (const iconnector& other)
{
	jacobians_ = other.jacobians_;
	jacobian_correction(&other);
	// this copies other's dependencies so, this and other share a graph
	if (g_man_) g_man_->suicide(this);
	g_man_ = graph_manager::get(const_cast<iconnector*>(&other), this);
}

void iconnector::move_helper (iconnector&& other)
{
	jacobians_ = std::move(other.jacobians_);
	jacobian_correction(&other);
	// this copies other's dependencies so, this and other share a graph
	if (g_man_)
	{
		g_man_->suicide(this);
	}
	g_man_ = graph_manager::get(&other, this);
	if (other.g_man_)
	{
		other.g_man_->suicide(&other);
		other.g_man_ = nullptr;
	}
}

void iconnector::jacobian_correction (const inode* other)
{
	// todo: move this down to immutable,
	// since if mutable, parent can have existing jacobian_ with references to other
	// assert this node has no parent (true when copying immutables)

	// check other's jacobians leafset for references to other and set to this
	for (auto& jpair : jacobians_)
	{
		std::list<std::pair<JTRANSFER,inode*>>& js = jpair.second.list_;
		if (js.back().second == other)
		{
			js.back().second = this;
		}
	}
}

}

#endif