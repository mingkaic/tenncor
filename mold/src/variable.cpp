//
//  variable.cpp
//  mold
//

#include "mold/variable.hpp"
#include "mold/functor.hpp"
#include "mold/error.hpp"

#ifdef MOLD_VARIABLE_HPP

namespace mold
{

bool Variable::has_data (void) const
{
	return nullptr != data_;
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
	notify_init();
}

void Variable::assign (const mold::iSource& src)
{
	if (nullptr == data_)
	{
		throw UninitializedError();
	}
	clay::State dest = data_->get_state();
	src.write_data(dest);
	for (iObserver* aud : audience_)
	{
		aud->update();
	}
}

void Variable::notify_init (void)
{
	for (iObserver* aud : audience_)
	{
		aud->initialize();
	}
}

}

#endif
