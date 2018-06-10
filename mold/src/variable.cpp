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

Variable::Variable (const Variable& other)
{
	if (nullptr != other.data_)
	{
		data_ = clay::TensorPtrT(other.data_->clone());
	}
}

Variable& Variable::operator = (const Variable& other)
{
	if (&other != this)
	{
		if (nullptr != other.data_)
		{
			data_ = clay::TensorPtrT(other.data_->clone());
		}
		else
		{
			data_ = nullptr;
		}
	}
	return *this;
}

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

void Variable::initialize (clay::TensorPtrT data)
{
	if (nullptr == data)
	{
		throw NilDataError();
	}
	data_ = std::move(data);
	for (iObserver* aud : audience_)
	{
		aud->initialize();
	}
}

iNode* Variable::clone_impl (void) const
{
	return new Variable(*this);
}

}

#endif
