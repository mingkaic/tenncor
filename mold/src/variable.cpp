//
//  variable.cpp
//  mold
//

#include "mold/variable.hpp"
#include "mold/functor.hpp"

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
		throw std::exception(); // todo: add context
	}
	return data_->get_state();
}

bool Variable::initialize (const clay::iBuilder& builder)
{
	auto out = builder.get();
	bool success = nullptr != out;
	if (success)
	{
		data_ = std::move(out);
		notify_init();
	}
	return success;
}

bool Variable::initialize (const clay::iBuilder& builder, clay::Shape shape)
{
	auto out = builder.get(shape);
	bool success = nullptr != out;
	if (success)
	{
		data_ = std::move(out);
		notify_init();
	}
	return success;
}

void Variable::assign (const clay::iSource& src)
{
	if (nullptr == data_)
	{
		throw std::exception(); // todo: add context
	}
	data_->read_from(src);
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
