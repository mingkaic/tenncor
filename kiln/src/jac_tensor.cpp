//
//  jac_tensor.cpp
//  kiln
//

#include "kiln/jac_tensor.hpp"

#ifdef KILN_JAC_TENSOR_HPP

namespace kiln
{

JacTensor::JacTensor (clay::Shape innershape,
    clay::Shape outershape, clay::DTYPE dtype) :
    ten_(clay::concatenate(innershape, outershape), dtype),
    hidden_(innershape.rank()) {}

clay::State JacTensor::get_state (void) const
{
    return ten_.get_state();
}

clay::State JacTensor::get_state (size_t idx) const
{
    clay::Shape internal = internal_shape();
    clay::State out = ten_.get_state(internal.n_elems() * idx);
    out.shape_ = internal;
    return out;
}

clay::Shape JacTensor::get_shape (void) const
{
    return ten_.get_shape();
}

clay::Shape JacTensor::external_shape (void) const
{
    std::vector<size_t> slist = ten_.get_shape().as_list();
    return std::vector<size_t>(slist.begin() + hidden_, slist.end());
}

clay::DTYPE JacTensor::get_type (void) const
{
    return ten_.get_type();
}

size_t JacTensor::total_bytes (void) const
{
    return ten_.total_bytes();
}

clay::iTensor* JacTensor::clone_impl (void) const
{
    return new JacTensor(*this);
}

clay::Shape JacTensor::internal_shape (void) const
{
    std::vector<size_t> slist = ten_.get_shape().as_list();
    return std::vector<size_t>(slist.begin(), slist.begin() + hidden_);
}

}

#endif
