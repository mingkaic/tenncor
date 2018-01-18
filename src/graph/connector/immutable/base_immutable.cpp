//
//  base_immutable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-06-26.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/base_immutable.hpp"
#include "include/graph/connector/immutable/immutable.hpp"

#ifdef TENNCOR_BASE_IMMUTABLE_HPP

namespace nnet
{

base_immutable::~base_immutable (void) { if (data_) delete data_; }

base_immutable* base_immutable::clone (void) const
{
	return static_cast<base_immutable*>(this->clone_impl());
}

base_immutable* base_immutable::move (void)
{
	return static_cast<base_immutable*>(this->move_impl());
}

base_immutable& base_immutable::operator = (const base_immutable& other)
{
	if (this != &other)
	{
		iconnector::operator = (other);
		copy_helper(other);
	}
	return *this;
}

base_immutable& base_immutable::operator = (base_immutable&& other)
{
	if (this != &other)
	{
		iconnector::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

varptr base_immutable::derive (inode* wrt)
{
	varptr out;
	iconnector* conn = dynamic_cast<iconnector*>(wrt);
	// check self
	if (wrt == this)
	{
		out = constant::get_shared_one();
	}
	// check cache
	else if (variable* leaf = dynamic_cast<variable*>(wrt))
	{
		out = this->get_gradient(leaf);
		// modify res with jacobians
		out = this->jacobian_call(out, leaf);
	}
	// check graph
	else if (conn && this->is_same_graph(conn))
	{
		// WARNING: this is one of the more expensive operations
		inode* temp_out = nullptr;
		this->temporary_eval(conn, temp_out);
		out = temp_out;
		// todo: apply jacobian (and test)

	}
	// is zero
	else
	{
		out = constant::get_shared_zero();
	}
	return out;
}

void base_immutable::temporary_eval (const iconnector* target, inode*& out) const
{
	constant* base = nullptr;
	out = temp_eval_helper(target, base);
	if (iconnector* outcon = dynamic_cast<iconnector*>(out))
	{
		outcon->update(std::unordered_set<size_t>{});
	}
}

tensorshape base_immutable::get_shape (void) const
{
	if (this->g_man_) this->g_man_->update();
	if (nullptr == data_)
	{
		return tensorshape();
	}
	return get_eval()->get_shape();
}

std::unordered_set<ileaf*> base_immutable::get_leaves (void) const
{
	std::unordered_set<ileaf*> leaves;
	for (auto leaf : gcache_)
	{
		leaves.emplace(leaf.first);
	}
	return leaves;
}

bool base_immutable::good_status (void) const
{
	return data_ != nullptr && data_->is_alloc();
}

bool base_immutable::read_proto (const tenncor::tensor_proto&)
{
	// it doesn't really make sense to deserialize data_ here, since data serves as a cache...
	return false;
}

void base_immutable::update (std::unordered_set<size_t>)
{
	bool allgood = true;
	bool damaged = false;
	for (size_t i = 0, n_subs = this->dependencies_.size();
		i < n_subs && allgood && !damaged; i++)
	{
		if (inode* a = dynamic_cast<inode*>(this->dependencies_[i]))
		{
			allgood = a->good_status() && allgood;
		}
		else
		{
			damaged = true;
		}
	}

	if (damaged)
	{
		// self destroy
		this->notify(UNSUBSCRIBE);
	}
	else if (allgood)
	{
		assert(this->g_man_);
		if (this->g_man_->freeze_ || 1 < this->dependencies_.size())
		// n-aries are pull update
		{
			this->g_man_->add_update(this,
			[this]
			{
				forward_pass();
			});
		}
		else
		// unaries are push update
		{
			// forward pass
			forward_pass();
			this->notify(UPDATE);
		}
	}
}

base_immutable::base_immutable (std::vector<inode*> args, std::string label) :
	iconnector(args, label)
{
	std::unordered_set<ileaf*> leafset;
	for (subject* sub : this->dependencies_)
	{
		std::unordered_set<ileaf*> leef = static_cast<inode*>(sub)->get_leaves();
		leafset.insert(leef.begin(), leef.end());
	}
	for (ileaf* l : leafset)
	{
		gcache_[l] = nullptr;
	}
}

base_immutable::base_immutable (const base_immutable& other) :
	iconnector(other)
{
	copy_helper(other);
}

base_immutable::base_immutable (base_immutable&& other) :
	iconnector(std::move(other))
{
	move_helper(std::move(other));
}

void base_immutable::death_on_broken (void)
{
	delete this;
}

const itensor* base_immutable::get_eval (void) const
{
	return data_;
}

inode* base_immutable::get_gradient (variable* leaf)
{
	auto it = gcache_.find(leaf);
	if (gcache_.end() == it)
	{
		return constant::get_shared_zero();
	}
	if (nullptr == it->second)
	{
		backward_pass(leaf);
	}
	return gcache_[leaf];
}

void base_immutable::copy_helper (const base_immutable& other)
{
	if (data_)
	{
		delete data_;
		data_ = nullptr;
	}
	if (other.data_)
	{
		data_ = other.data_->clone();
	}
	gcache_ = other.gcache_;
}

void base_immutable::move_helper (base_immutable&& other)
{
	if (data_)
	{
		delete data_;
	}
	data_ = std::move(other.data_);
	other.data_ = nullptr;
	gcache_ = std::move(other.gcache_);
}

inode* base_immutable::temp_eval_helper (const iconnector* target, constant*& base) const
{
	// base case
	if (this == target)
	{
		// return 1
		if (!base)
		{
			base = constant::get(1);
		}
		return base;
	}
	// traverse towards target by comparing leaf sets
	std::vector<inode*> args;
	for (subject* sub : this->dependencies_)
	{
		inode* arg = static_cast<inode*>(sub);
		base_immutable* con = dynamic_cast<base_immutable*>(arg);
		if (nullptr != con && con->potential_descendent(target))
		{
			args.push_back(con->temp_eval_helper(target, base));
		}
		else
		{
			args.push_back(arg);
		}
	}
	// create a new copy of this with out sharing base's life cycle
	base_immutable* base_imm = this->arg_clone(args);
	base_imm->add_ondeath_dependent(base);
	return base_imm;
}

}

#endif