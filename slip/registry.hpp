/*!
 *
 *  registry.hpp
 *  slip
 *
 *  Purpose:
 *  generic operation registry
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/functor.hpp"

#include "slip/opcode.hpp"

#pragma once
#ifndef SLIP_REGISTRY_HPP
#define SLIP_REGISTRY_HPP

namespace slip
{

bool has_op (OPCODE opcode);

mold::OperatePtrT get_op (OPCODE opcode);

}

#endif /* SLIP_REGISTRY_HPP */
