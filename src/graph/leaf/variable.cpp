//
//  variable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-27.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/variable.hpp"

#ifdef TENNCOR_VARIABLE_HPP

namespace nnet
{

variable::variable (const tensorshape& shape,
	std::shared_ptr<data_src> source,
	std::string label) :
inode(label), src_(source), data_(new tensor(shape)) {}

variable::variable (const variable& other) :
	inode(other)
{
	copy_helper(other);
}

variable::variable (variable&& other) :
	inode(std::move(other))
{
	move_helper(std::move(other));
}

variable* variable::clone (void) const
{
	return static_cast<variable*>(this->clone_impl());
}

variable* variable::move (void)
{
	return static_cast<variable*>(this->move_impl());
}

variable& variable::operator = (const variable& other)
{
	if (this != &other)
	{
		inode::operator = (other);
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

variable& variable::operator = (variable&& other)
{
	if (this != &other)
	{
		inode::operator = (std::move(other));
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}


tensor* variable::get_tensor (void)
{
	return data_.get();
}

varptr variable::derive (inode* wrt)
{
	constant* out = nullptr;
	if (this == wrt)
	{
		tensorshape shape = data_->get_shape();
		std::vector<double> data(shape.n_elems(), 1); // change to match wrt type
		out = constant::get<double>(data, shape);
	}
	return out;
}

bool variable::initialize (void)
{
	bool success = data_->read_from(*src_);
	if (success)
	{
		this->notify(UPDATE);
	}
	return success;
}

bool variable::initialize (tensorshape shape)
{
	bool success = data_->read_from(*src_, shape);
	if (success)
	{
		this->notify(UPDATE);
	}
	return success;
}

bool variable::assign (inode* input, bool notify)
{
	bool successful = input != this && input != nullptr;
	if (successful)
	{
		tensor* itens = input->get_tensor();
		successful = itens->has_data() && itens->is_compatible_with(*data_);
		if (successful)
		{
			assign_io asgn;
			itens->write_to(asgn);
			data_->read_from(asgn);
			if (notify)
			{
				this->notify(UPDATE);
			}
		}
	}
	return successful;
}


NODE_TYPE variable::node_type (void) const
{
	return VARIABLE_T;
}

void variable::serialize_detail (google::protobuf::Any* proto_dest)
{
	tenncor::variable_proto var;
	std::vector<size_t> slist = data_->get_allowed().as_list();
	google::protobuf::RepeatedField<uint64_t> shape_field(slist.begin(), slist.end());
	var.mutable_allowed_shape()->Swap(&shape_field);

	tenncor::source_proto src_dest;
	src_->serialize(src_dest);
	var.mutable_source()->Swap(&src_dest);

	var.set_locpos(data_ep_);

	proto_dest->PackFrom(var);
}


inode* variable::clone_impl (void) const
{
	return new variable(*this);
}

inode* variable::move_impl (void)
{
	return new variable(std::move(*this));
}

void variable::copy_helper (const variable& other)
{
	if (nullptr == other.data_)
	{
		data_ = nullptr;
	}
	else
	{
		data_ = std::make_unique<tensor>(*other.data_);
	}

	if (nullptr == other.src_)
	{
		src_ = nullptr;
	}
	else
	{
		src_ = std::shared_ptr<data_src>(other.src_->clone());
	}
}

void variable::move_helper (variable&& other)
{
	data_ = std::move(other.data_);
	src_ = std::move(other.src_);
}

}

#endif
