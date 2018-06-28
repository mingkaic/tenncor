//
//  operate_io.cpp
//  slip
//

#include "slip/include/operate_io.hpp"

#ifdef SLIP_OPERATE_IO_HPP

namespace slip
{

OperateIO::OperateIO (TypeRegT ops, ShaperF shaper, TyperF typer) :
    ops_(ops), shaper_(shaper), typer_(typer) {}

bool OperateIO::validate_data (clay::State state,
    std::vector<mold::StateRange> args) const
{
    auto imms = get_imms(args);
    return state.shape_.is_compatible_with(imms.first) &&
        state.dtype_ == imms.second;
}

bool OperateIO::write_data (clay::State& dest,
    std::vector<mold::StateRange> args) const
{
    auto imms = get_imms(args);
    bool success = dest.shape_.
        is_compatible_with(imms.first) &&
        dest.dtype_ == imms.second;
    if (success)
    {
        unsafe_write(dest, args, imms.second);
    }
    return success;
}

clay::TensorPtrT OperateIO::make_data (
    std::vector<mold::StateRange> args) const
{
    auto imms = get_imms(args);
    clay::Shape& shape = imms.first;
    clay::DTYPE& dtype = imms.second;
    clay::Tensor* out = new clay::Tensor(shape, dtype);
    clay::State dest = out->get_state();
    unsafe_write(dest, args, dtype);
    return clay::TensorPtrT(out);
}

ImmPair OperateIO::get_imms (std::vector<mold::StateRange>& args) const
{
    if (args.empty())
    {
        throw NoArgumentsError();
    }
    std::vector<clay::DTYPE> types(args.size());
    std::transform(args.begin(), args.end(), types.begin(),
    [](mold::StateRange& state) -> clay::DTYPE
    {
        return state.type();
    });
    clay::DTYPE otype = typer_(types);
    return {shaper_(args), otype};
}

void OperateIO::unsafe_write (clay::State& dest,
    std::vector<mold::StateRange>& args, clay::DTYPE dtype) const
{
    auto op = ops_.find(dtype);
    if (ops_.end() == op)
    {
        throw clay::UnsupportedTypeError(dtype);
    }
    std::vector<clay::State> states(args.size());
    std::transform(args.begin(), args.end(), states.begin(),
    [](const mold::StateRange& sr) -> clay::State
    {
        return sr.arg_;
    });
    op->second(dest, states);
}

mold::iOperateIO* OperateIO::clone_impl (void) const
{
    return new OperateIO(*this);
}

}

#endif
