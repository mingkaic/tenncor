//
//  variable.cpp
//  mold
//

#include "mold/variable.hpp"
#include "mold/iobserver.hpp"
#include "mold/error.hpp"

#ifdef MOLD_VARIABLE_HPP

namespace mold
{
	
Variable::Variable (void) = default;

Variable::Variable (const Variable& other)
{
	if (nullptr != other.data_)
	{
		data_ = std::make_unique<clay::Tensor>(*other.data_);
	}
}

Variable::Variable (Variable&&) = default;

Variable& Variable::operator = (const Variable& other)
{
	if (&other != this)
	{
		if (nullptr != other.data_)
		{
			data_ = std::make_unique<clay::Tensor>(*other.data_);
		}
		else
		{
			data_ = nullptr;
		}
	}
	return *this;
}

Variable& Variable::operator = (Variable&&) = default;

bool Variable::has_data (void) const
{
	return nullptr != data_;
}

clay::Shape Variable::get_shape (void) const
{
	return data_->get_shape();
}

clay::State Variable::get_state (void) const
{
	if (nullptr == data_)
	{
		throw UninitializedError();
	}
	return data_->get_state();
}

void Variable::set_data (clay::TensorPtrT data)
{
	if (nullptr == data)
	{
		throw NilDataError();
	}
	if (nullptr != data_)
	{
		throw std::exception(); // todo: add context reinitializing
	}
	data_ = std::move(data);
}

iNode* Variable::clone_impl (void) const
{
	return new Variable(*this);
}

}

#endif
