// 
//  functor.cpp 
//  cnnet 
// 
//  Created by Mingkai Chen on 2016-12-01. 
//  Copyright © 2018 Mingkai Chen. All rights reserved. 
//

#include "include/graph/func/functor.hpp"

#ifdef TENNCOR_FUNCTOR_HPP

namespace nnet
{

#define STR2CODE(CODE) std::pair<std::string,OPCODE>{#CODE, CODE},

static std::unordered_map<std::string,OPCODE> code_map =
{
	STR2CODE(ABS)
	STR2CODE(NEG)
	STR2CODE(NOT)
	STR2CODE(SIN)
	STR2CODE(COS)
	STR2CODE(TAN)
	STR2CODE(CSC)
	STR2CODE(SEC)
	STR2CODE(COT)
	STR2CODE(EXP)
	STR2CODE(LOG)
	STR2CODE(SQRT)
	STR2CODE(ROUND)
	STR2CODE(POW)
	STR2CODE(ADD)
	STR2CODE(SUB)
	STR2CODE(MUL)
	STR2CODE(DIV)
	STR2CODE(EQ)
	STR2CODE(NE)
	STR2CODE(GT)
	STR2CODE(LT)
	STR2CODE(BINO)
	STR2CODE(UNIF)
	STR2CODE(NORM)
	STR2CODE(TRANSPOSE)
	STR2CODE(FLIP)
	STR2CODE(ARGMAX)
	STR2CODE(MAX)
	STR2CODE(SUM)
	STR2CODE(EXPAND)
	STR2CODE(N_ELEMS)
	STR2CODE(N_DIMS)
	STR2CODE(MATMUL)
	STR2CODE(INJACOBIAN)
	STR2CODE(OUTJACOBIAN)
	STR2CODE(JACOBIANLEFT)
	STR2CODE(JACOBIANRIGHT)
};

static std::vector<std::string> code_vec = ([](void)
{
	assert(code_map.size() == _END_SENTINEL);
	std::vector<std::string> out(_END_SENTINEL);
	for (auto& code_pair : code_map)
	{
		out[code_pair.second] = code_pair.first;
	}
	return out;
})();

functor* functor::get (std::vector<inode*> args, TENSOP_F tensop, 
	DERIVE_F derive, OPCODE code)
{
	assert(false == args.empty());
	return new functor(args, tensop, derive, code);
}

functor::~functor (void) {}

functor* functor::clone (void) const
{
	return static_cast<functor*>(this->clone_impl());
}

functor* functor::move (void)
{
	return static_cast<functor*>(this->move_impl());
}

functor& functor::operator = (const functor& other)
{
	if (this != &other)
	{
		iobserver::operator = (other);
		inode::operator = (other);
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

functor& functor::operator = (functor&& other)
{
	if (this != &other)
	{
		iobserver::operator = (std::move(other));
		inode::operator = (std::move(other));
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}

std::string functor::get_name (void) const
{
	std::string args;
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

std::unordered_set<const inode*> functor::get_leaves (void) const
{
	std::unordered_set<const inode*> leaves;
	std::vector<inode*> args = this->get_arguments();
	for (inode* arg : args)
	{
		std::unordered_set<const inode*> subleaves = arg->get_leaves();
		leaves.insert(subleaves.begin(), subleaves.end());
	}
	return leaves;
}

std::vector<inode*> functor::get_arguments (void) const
{
	std::vector<inode*> node_args(this->dependencies_.size());
	std::transform(this->dependencies_.begin(), this->dependencies_.end(), node_args.begin(),
		[](subject* s) { return static_cast<inode*>(s); });
	return node_args;
}

tensor* functor::get_tensor (void)
{
	return data_.get();
}

varptr functor::derive (inode* wrt)
{
	if (wrt == this)
	{
		tensor* ten = wrt->get_tensor();
		assert(ten && ten->has_data());
		tensorshape shape = ten->get_shape();
		std::vector<double> data(shape.n_elems(), 1); // change to match wrt type
		return constant::get<double>(data, shape);
	}
	if (nullptr == data_)
	{
		throw std::exception(); // uninitialized variables
	}
	return derive_(wrt, get_arguments());
}


NODE_TYPE functor::node_type (void) const
{
	return FUNCTOR_T;
}

void functor::serialize_detail (google::protobuf::Any* proto_dest)
{
	tenncor::functor_proto func_dest;
	func_dest.set_opcode(opcode_);
	std::vector<inode*> args = get_arguments();
	for (inode* arg : args)
	{
		func_dest.add_args(arg->get_uid()); 
	}
	proto_dest->PackFrom(func_dest);
}


functor::functor (std::vector<inode*> args, TENSOP_F tensop, DERIVE_F derive, OPCODE code) :
	inode(code_vec[code]), iobserver(std::vector<subject*>(args.begin(), args.end())),
	tensop_(tensop), derive_(derive), opcode_(code)
{ this->update(); }

functor::functor (const functor& other) : 
	inode(other), iobserver(other)
{ copy_helper(other); }

functor::functor (functor&& other) : 
	inode(std::move(other)), iobserver(std::move(other))
{ move_helper(std::move(other)); }

inode* functor::clone_impl (void) const
{
	return new functor(*this);
}

inode* functor::move_impl (void)
{
	return new functor(std::move(*this));
}

void functor::copy_helper (const functor& other)
{
	tensop_ = other.tensop_;
	derive_ = other.derive_;
	if (nullptr != other.io_)
	{
		io_ = std::unique_ptr<idata_src>(other.io_->clone());
	}
	if (nullptr != other.data_)
	{
		data_ = std::make_unique<tensor>(*other.data_);
	}
}

void functor::move_helper (functor&& other)
{
	tensop_ = std::move(other.tensop_);
	derive_ = std::move(other.derive_);
	io_ = std::move(other.io_);
	data_ = std::move(other.data_);
}

void functor::update (void)
{
	std::vector<inode*> args = get_arguments();
	bool has_data = std::all_of(args.begin(), args.end(), 
	[](inode* node)
	{
		tensor* tens = node->get_tensor();
		return nullptr != tens && tens->has_data();
	});
	if (has_data)
	{
		if (nullptr == data_)
		{
			data_ = std::unique_ptr<tensor>(tensop_(io_, args));
		}
		data_->read_from(*io_);
		this->notify(UPDATE);
	}
}

void functor::death_on_broken (void)
{
	delete this;
}

}

#endif
