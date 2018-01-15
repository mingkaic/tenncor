//
// Created by Mingkai Chen on 2017-07-03.
//

#include "include/graph/connector/immutable/shape_dep.hpp"

#ifdef TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

shape_dep::~shape_dep (void)
{
	delete shape_info;
}

shape_dep* shape_dep::get (std::vector<inode*> args,
	SHAPE_EXTRACT forward, tensorshape shape, std::string name)
{
	size_t n_args = args.size();
	assert(n_args> 0);
	std::unordered_set<inode*> audience;
	if (args[0]->find_audience(name, audience))
	{
		// share nodes when possible
		for (inode* aud : audience)
		{
			if (shape_dep* saud = dynamic_cast<shape_dep*>(aud))
			{
				std::vector<inode*> aud_args = aud->get_arguments();
				if (n_args == aud_args.size())
				{
					bool all_aud = true;
					for (size_t i = 0; i < n_args; i++)
					{
						all_aud = all_aud && args[i] == aud_args[i];
					}
					if (all_aud) return saud;
				}
			}
		}
	}
	return new shape_dep(args, forward, shape, name);
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
		shape_info = other.shape_info->clone();
	}
	return *this;
}

shape_dep& shape_dep::operator = (shape_dep&& other)
{
	if (this != &other)
	{
		base_immutable::operator = (std::move(other));
		shape_ = std::move(other.shape_);
		shape_info = other.shape_info->move();
		other.shape_info = nullptr;
	}
	return *this;
}

shape_dep::shape_dep (std::vector<inode*> args,
	SHAPE_EXTRACT forward, tensorshape shape, std::string label) :
base_immutable(args, label),
shape_info(new shape_extracter<double>(forward)),
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
	shape_info = other.shape_info->clone();
}

shape_dep::shape_dep (shape_dep&& other) :
	base_immutable(std::move(other))
{
	shape_ = std::move(other.shape_);
	shape_info = other.shape_info->move();
	other.shape_info = nullptr;
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
	return new shape_dep(args, shape_info->get_shaper(), shape_, this->get_label());
}

void shape_dep::forward_pass (void)
{
	std::vector<tensorshape> shapes;
	for (subject* sub : this->dependencies_)
	{
		shapes.push_back(this->take_eval(static_cast<inode*>(sub))->get_shape());
	}
	if (nullptr == this->data_)
	{
		this->data_ = new tensor<double>(shape_);
	}
	(*shape_info)(*this->data_, shapes);
}

void shape_dep::backward_pass (variable* leaf)
{
	this->gcache_[leaf] = nnet::constant::get_shared_zero();
}

}

#endif