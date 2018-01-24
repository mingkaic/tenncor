//
// Created by Mingkai Chen on 2017-07-03.
//

#include "include/graph/connector/immutable/shape_dep.hpp"

#ifdef TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

shape_dep::~shape_dep (void){}

shape_dep* shape_dep::get (inode* arg, SHAPE_EXTRACT forward, 
	tensorshape shape, std::string name)
{
	std::unordered_set<inode*> audience;
	if (arg->find_audience(name, audience))
	{
		// share nodes when possible
		for (inode* aud : audience)
		{
			if (shape_dep* saud = dynamic_cast<shape_dep*>(aud))
			{
				std::vector<inode*> aud_args = aud->get_arguments();
				if (1 == aud_args.size() && arg == aud_args[0])
				{
					return saud;
				}
			}
		}
	}
	return new shape_dep(arg, forward, shape, name);
}

shape_dep* shape_dep::clone (void) const
{
	return static_cast<shape_dep*>(this->clone_impl());
}

shape_dep* shape_dep::move (void)
{
	return static_cast<shape_dep*>(this->move_impl());
}

shape_dep& shape_dep::operator = (const shape_dep& other)
{
	if (this != &other)
	{
		base_immutable::operator = (other);
		shape_ = other.shape_;
		assigner_ = other.assigner_;
		extracter_ = other.extracter_;
	}
	return *this;
}

shape_dep& shape_dep::operator = (shape_dep&& other)
{
	if (this != &other)
	{
		base_immutable::operator = (std::move(other));
		shape_ = std::move(other.shape_);
		assigner_ = std::move(other.assigner_);
		extracter_ = std::move(other.extracter_);
	}
	return *this;
}

shape_dep::shape_dep (inode* arg, SHAPE_EXTRACT forward, 
	tensorshape shape, std::string label) :
base_immutable({arg}, label),
extracter_(forward),
shape_(shape)
{
	shape_.assert_is_fully_defined();
	this->jacobians_.clear();
	this->update(std::unordered_set<size_t>{});
}

shape_dep::shape_dep (const shape_dep& other) :
	base_immutable(other)
{
	shape_ = other.shape_;
	assigner_ = other.assigner_;
	extracter_ = other.extracter_;
}

shape_dep::shape_dep (shape_dep&& other) :
	base_immutable(std::move(other))
{
	shape_ = std::move(other.shape_);
	assigner_ = std::move(other.assigner_);
	extracter_ = std::move(other.extracter_);
}

inode* shape_dep::clone_impl (void) const
{
	return new shape_dep(*this);
}

inode* shape_dep::move_impl (void)
{
	return new shape_dep(std::move(*this));
}

base_immutable* shape_dep::arg_clone (std::vector<inode*> args) const
{
	return new shape_dep(args[0], extracter_, shape_, this->get_label());
}

void shape_dep::forward_pass (void)
{
	inode* node = static_cast<inode*>(this->dependencies_[0]);
	tenncor::tensor_proto::tensor_t type = node->get_type();
	if (nullptr == this->data_)
	{
		switch (type)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				this->data_ = new tensor_double(shape_);
			break;
			case tenncor::tensor_proto::SIGNED_T:
				this->data_ = new tensor_signed(shape_);
			break;
			default:
				throw std::exception(); // unsupported type
		}
	}
	tensorshape shape = this->take_eval(node)->get_shape();
	std::vector<size_t> tsvec = extracter_(shape);
	switch (type)
	{
		case tenncor::tensor_proto::DOUBLE_T:
		{
			assigner_(*(this->data_), 
				&std::vector<double>(tsvec.begin(), tsvec.end())[0], type);
		}
		break;
		case tenncor::tensor_proto::SIGNED_T:
		{
			assigner_(*(this->data_), 
				&std::vector<signed>(tsvec.begin(), tsvec.end())[0], type);
		}
		break;
		default:
		break;
	}
}

void shape_dep::backward_pass (variable* leaf)
{
	this->gcache_[leaf] = nnet::constant::get_shared_zero();
}

}

#endif
