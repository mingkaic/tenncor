//
//  functor.cpp
//  mold
//

#include "clay/memory.hpp"

#include "mold/operate_io.hpp"

#ifdef MOLD_OPERATE_IO_HPP

namespace mold
{

OperateIO::OperateIO (ArgsF op, ShaperF shaper, TyperF typer) :
    op_(op), shaper_(shaper), typer_(typer) {}

bool OperateIO::read_data (clay::State& dest) const
{
    auto outs = expect_out();
    bool success = dest.shape_.is_compatible_with(outs.first) &&
        dest.dtype_ == outs.second;
    if (success)
    {
        op_(dest, args_);
    }
    return success;
}

clay::TensorPtrT OperateIO::get (void) const
{
    auto outs = expect_out();
    size_t nbytes = outs.first.n_elems() * clay::type_size(outs.second);
    std::shared_ptr<char> data = clay::make_char(nbytes);
    clay::TensorPtrT ptr(new clay::Tensor(data, outs.first, outs.second));
    clay::State dest(data, outs.first, outs.second);
    op_(dest, args_);
    return ptr;
}

std::pair<clay::Shape, clay::DTYPE> OperateIO::expect_out (void) const
{
    std::vector<clay::Shape> shapes;
    std::vector<clay::DTYPE> types;
    for (const clay::State& arg : args_)
    {
        shapes.push_back(arg.shape_);
        types.push_back(arg.dtype_);
    }
    return {shaper_(shapes), typer_(types)};
}

}

#endif
