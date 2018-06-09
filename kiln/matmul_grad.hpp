/*!
 *
 *  matmul_grad.hpp
 *  kiln
 *
 *  Purpose:
 *  matmul gradient function
 *
 *  Created by Mingkai Chen on 2016-10-24.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "kiln/identifier.hpp"
#include "kiln/functor.hpp"

#pragma once
#ifndef KILN_MATMUL_GRAD_HPP
#define KILN_MATMUL_GRAD_HPP

namespace kiln
{

Identifier* matmul_grad (Identifier*, GradArgsT args);

}

#endif /* KILN_MATMUL_GRAD_HPP */
