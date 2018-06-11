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

#include "clay/tensor.hpp"

#pragma once
#ifndef KILN_JAC_TENSOR_HPP
#define KILN_JAC_TENSOR_HPP

namespace kiln
{

class JacTensor final : public clay::iTensor
{
public:
    JacTensor (clay::Shape innershape,
        clay::Shape outershape, clay::DTYPE dtype);

	//! get internal state
	clay::State get_state (void) const override;

    clay::State get_state (size_t idx) const override;

	//! get tensor shape
	clay::Shape get_shape (void) const override;

    clay::Shape external_shape (void) const;

	//! get tensor dtype
    clay::DTYPE get_type (void) const override;

	//! get bytes allocated
    size_t total_bytes (void) const override;

private:
    clay::iTensor* clone_impl (void) const override;

    clay::Shape internal_shape (void) const;

    clay::Tensor ten_;

    size_t hidden_;
};

}

#endif /* KILN_JAC_TENSOR_HPP */
