//
//  variable.cpp
//  mold
//

#include "mold/variable.hpp"

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

iNode* Variable::derive (iNode* wrt)
{
	if (data_ == nullptr)
	{
		throw std::exception(); // todo: add context
	}
	iNode* out;
	clay::DTYPE otype = data_->get_type();
	if (this == wrt)
	{
		out = make_one(otype);
	}
	else
	{
		unsigned short bsize = clay::type_size(otype);
		std::shared_ptr<char> data = clay::make_char(bsize);
		memset(data.get(), 0, bsize);
		out = new Constant(data, clay::Shape(std::vector<size_t>{1}), otype); 
	}
	if (nullptr == out)
	{
		throw std::exception(); // todo: add context
	}
	return out;
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
