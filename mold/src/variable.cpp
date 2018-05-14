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
    return nullptr == data_;
}

clay::State Variable::get_state (void) const
{
    if (nullptr == data_)
    {
        throw std::exception(); // todo: add context
    }
    return data_->get_state();
}

NodePtrT Variable::derive (NodeRefT wrt)
{
    NodePtrT out;
    switch (data_->get_type())
    {
        case clay::DTYPE::DOUBLE:
            out = make_constant((double) 1);
        break;
        case clay::DTYPE::FLOAT:
            out = make_constant((float) 1);
        break;
        case clay::DTYPE::INT8:
            out = make_constant((int8_t) 1);
        break;
        case clay::DTYPE::INT16:
            out = make_constant((int16_t) 1);
        break;
        case clay::DTYPE::INT32:
            out = make_constant((int32_t) 1);
        break;
        case clay::DTYPE::INT64:
            out = make_constant((int64_t) 1);
        break;
        case clay::DTYPE::UINT8:
            out = make_constant((uint8_t) 1);
        break;
        case clay::DTYPE::UINT16:
            out = make_constant((uint16_t) 1);
        break;
        case clay::DTYPE::UINT32:
            out = make_constant((uint32_t) 1);
        break;
        case clay::DTYPE::UINT64:
            out = make_constant((uint64_t) 1);
        break;
        default:
            throw std::exception();
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
    data_->read_from(src);
    for (Functor* aud : audience_)
    {
        aud->update();
    }
}

void Variable::notify_init (void)
{
    for (Functor* aud : audience_)
    {
        aud->initialize();
    }
}

}

#endif
