/*!
 *
 *  matmul_grad.hpp
 *  wire
 *
 *  Purpose:
 *  matmul gradient function
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "wire/identifier.hpp"
#include "wire/functor.hpp"

#pragma once
#ifndef WIRE_MATMUL_GRAD_HPP
#define WIRE_MATMUL_GRAD_HPP

namespace wire
{

Identifier* matmul_grad (Identifier*, GradArgsT args);

}

#endif /* WIRE_MATMUL_GRAD_HPP */
