/*!
 *
 *  jac_tensor.hpp
 *  kiln
 *
 *  Purpose:
 *  jacobian tensor implementation (treating tensors as a type)
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/itensor.hpp"

#pragma once
#ifndef KILN_JAC_TENSOR_HPP
#define KILN_JAC_TENSOR_HPP

namespace kiln
{

class JacTensor final : public clay::iTensor
{
public:
    JacTensor (clay::Shape innershape,
        clay::Shape outershape, clay::DTYPE dtype) :
        ten_(clay::concatenate(innershape, outershape), dtype),
        hidden_(innershape.rank()) {}

	//! get internal state
	clay::State get_state (void) const override
    {
        return ten_.get_state();
    }

    clay::State get_State (size_t idx) const override
    {
        clay::Shape internal = get_internal();
        clay::State out = ten_.get_state(internal.n_elems() * idx);
        out.shape_ = internal;
        return out;
    }

	//! get tensor shape
	clay::Shape get_shape (void) const override
    {
        return ten_.get_shape();
    }

    clay::Shape external_shape (void) const
    {
        std::vector<size_t> slist = ten_.get_shape().as_list();
        return std::vector<size_t>(slist.begin() + hidden_, slist.end());
    }

	//! get tensor dtype
    clay::DTYPE get_type (void) const override
    {
        return ten_.get_type();
    }

	//! get bytes allocated
    size_t total_bytes (void) const override
    {
        return ten_.total_bytes();
    }

private:
    iTensor* clone_impl (void) const override
    {
        return new JacTensor(*this);
    }

    clay::Shape internal_shape (void) const
    {
        std::vector<size_t> slist = ten_.get_shape().as_list();
        return std::vector<size_t>(slist.begin(), slist.begin() + hidden_);
    }

    clay::Tensor ten_;

    size_t hidden_;
}

}

#endif /* KILN_JAC_TENSOR_HPP */
