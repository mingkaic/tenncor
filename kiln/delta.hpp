/*!
 *
 *  delta.hpp
 *  kiln
 *
 *  Purpose:
 *  Create derivatives of a function
 *
 *  Created by Mingkai Chen on 2018-06-05.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "kiln/identifier.hpp"

#ifndef KILN_DELTA_HPP
#define KILN_DELTA_HPP

namespace kiln
{

Identifier* delta (Identifier* root, Identifier* wrt);

}

#endif /* KILN_DELTA_HPP */
