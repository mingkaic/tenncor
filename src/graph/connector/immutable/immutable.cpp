//
//  immutable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-06-26.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/immutable.hpp"
// #include "include/graph/connector/immutable/linear.hpp"

#ifdef TENNCOR_IMMUTABLE_HPP

namespace nnet
{

immutable::~immutable (void) { if (data_) delete data_; }

immutable* immutable::clone (void) const
{
	return static_cast<immutable*>(this->clone_impl());
}

immutable* immutable::move (void)
{
	return static_cast<immutable*>(this->move_impl());
}

immutable& immutable::operator = (const immutable& other)
{
	if (this != &other)
	{
		iconnector::operator = (other);
		copy_helper(other);
	}
	return *this;
}

immutable& immutable::operator = (immutable&& other)
{
	if (this != &other)
	{
		iconnector::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

varptr immutable::derive (inode* wrt)
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
// todo: deprecate this series of backpropagation once nlinear is implemented
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

void immutable::temporary_eval (const iconnector* target, inode*& out) const
{
	constant* base = nullptr;
	out = temp_eval_helper(target, base);
	if (iconnector* outcon = dynamic_cast<iconnector*>(out))
	{
		outcon->update(std::unordered_set<size_t>{});
	}
}

tensorshape immutable::get_shape (void) const
{
	if (this->g_man_) this->g_man_->update();
	if (nullptr == data_)
	{
		return tensorshape();
	}
	return get_eval()->get_shape();
}

std::unordered_set<ileaf*> immutable::get_leaves (void) const
{
	std::unordered_set<ileaf*> leaves;
	for (auto leaf : gcache_)
	{
		leaves.emplace(leaf.first);
	}
	return leaves;
}

bool immutable::good_status (void) const
{
	return data_ != nullptr && data_->is_alloc();
}

bool immutable::read_proto (const tenncor::tensor_proto&)
{
	// it doesn't really make sense to deserialize data_ here, since data serves as a cache...
	return false;
}

void immutable::update (std::unordered_set<size_t>)
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

immutable::immutable (std::vector<inode*> args, std::string label) :
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

immutable::immutable (const immutable& other) :
	iconnector(other)
{
	copy_helper(other);
}

immutable::immutable (immutable&& other) :
	iconnector(std::move(other))
{
	move_helper(std::move(other));
}

void immutable::death_on_broken (void)
{
	delete this;
}

const tensor* immutable::get_eval (void) const
{
	return data_;
}

inode* immutable::get_gradient (variable* leaf)
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

void immutable::copy_helper (const immutable& other)
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

void immutable::move_helper (immutable&& other)
{
	if (data_)
	{
		delete data_;
	}
	data_ = other.data_->move();
	other.data_ = nullptr;
	gcache_ = std::move(other.gcache_);
}

inode* immutable::temp_eval_helper (const iconnector* target, constant*& base) const
{
	// base case
	if (this == target)
	{
		// return 1
		if (!base)
		{
			// todo: get rid of switch once type conversion is implemented
			switch (target->get_type())
			{
				case BAD:
				// todo: resolve by type forward lookup
				case DOUBLE:
					base = constant::get((double) 1);
				break;
				case INT:
					base = constant::get((signed) 1);
				break;
				default:
					throw std::exception(); // unsupported type
			}
		}
		return base;
	}
	// traverse towards target by comparing leaf sets
	std::vector<inode*> args;
	for (subject* sub : this->dependencies_)
	{
		inode* arg = static_cast<inode*>(sub);
		immutable* con = dynamic_cast<immutable*>(arg);
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
	immutable* base_imm = this->arg_clone(args);
	base_imm->add_ondeath_dependent(base);
	return base_imm;
}

}

#endif