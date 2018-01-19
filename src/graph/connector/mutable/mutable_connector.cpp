//
//  mutable_connector.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-27.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/mutable/mutable_connector.hpp"

#ifdef mutable_connect_hpp

namespace nnet
{

void mutable_connector::connect (void)
{
	if (valid_args() && nullptr == ic_)
	{
		iconnector* con = dynamic_cast<iconnector*>(op_maker_(arg_buffers_));
		delete ic_;
		ic_ = con;
		this->add_dependency(con);
		ic_->set_death((void**) &ic_); // ic_ resets to nullptr when deleted
	}
}

void mutable_connector::disconnect (void)
{
	if (nullptr != ic_)
	{
		// severe our dependency on ic_
		// to prevent this from getting destroyed
		this->kill_dependencies();
		delete ic_;
	}
}

mutable_connector::mutable_connector (MAKE_CONNECT maker, size_t nargs) :
	iconnector(std::vector<inode*>{}, "mutable_connector"),
	op_maker_(maker), arg_buffers_(nargs, nullptr) {}

mutable_connector::mutable_connector (const mutable_connector& other) :
	iconnector(other), op_maker_(other.op_maker_),
	arg_buffers_(other.arg_buffers_) {}

mutable_connector* mutable_connector::get (MAKE_CONNECT maker, size_t nargs)
{
	return new mutable_connector(maker, nargs);
}

mutable_connector::~mutable_connector (void)
{
	if (nullptr != ic_)
	{
		delete ic_;
	}
}

mutable_connector* mutable_connector::clone (void)
{
	return new mutable_connector(*this);
}

mutable_connector& mutable_connector::operator = (const mutable_connector& other)
{
	if (&other != this)
	{
		iconnector::operator = (other);
		op_maker_ = other.op_maker_;
		arg_buffers_ = other.arg_buffers_;
	}
	return *this;
}

tensorshape mutable_connector::get_shape(void)
{
	if (nullptr == ic_)
	{
		return tensorshape();
	}
	return ic_->get_shape();
}

tensor<double>* mutable_connector::get_eval(void)
{
	if (nullptr == ic_)
	{
		return nullptr;
	}
	return ic_->get_eval();
}

varptr mutable_connector::derive(void)
{
	if (nullptr == ic_)
	{
		return nullptr;
	}
	return ic_->derive();
}

void mutable_connector::update (std::unordered_set<size_t> argidx)
{
	if (update_message::REMOVE_ARG == msg.cmd_)
	{
		disconnect();
	}
	else
	{
		this->notify(msg);
	}
}

bool mutable_connector::add_arg (inode* var, size_t idx)
{
	bool replace = nullptr != arg_buffers_[idx];
	arg_buffers_[idx] = var;
	if (replace)
	{
		disconnect();
	}
	connect();
	return replace;
}

bool mutable_connector::remove_arg (size_t idx)
{
	if (nullptr != arg_buffers_[idx])
	{
		disconnect();
		arg_buffers_[idx] = nullptr;
		return true;
	}
	return false;
}

bool mutable_connector::valid_args (void)
{
	for (varptr arg : arg_buffers_)
	{
		if (nullptr == arg.get())
		{
			return false;
		}
	}
	return true;
}

size_t mutable_connector::nargs (void) const
{
	return arg_buffers_.size();
}

void mutable_connector::get_args (std::vector<inode*>& args) const
{
	args.clear();
	for (varptr a : arg_buffers_)
	{
		args.push_back(a.get());
	}
}

}

#endif
