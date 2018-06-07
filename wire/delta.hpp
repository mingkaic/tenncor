/*!
 *
 *  delta.hpp
 *  wire
 *
 *  Purpose:
 *  Create derivatives of a function
 *
 *  Created by Mingkai Chen on 2018-06-05.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "wire/identifier.hpp"

#ifndef WIRE_DELTA_HPP
#define WIRE_DELTA_HPP

namespace wire
{

Identifier* delta (Identifier* root, Identifier* wrt);

}

#endif /* WIRE_DELTA_HPP */
