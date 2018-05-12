/*!
 *
 *  memory.hpp
 *  clay
 *
 *  Purpose:
 *  create a tensor
 *
 *  Created by Mingkai Chen on 2018-05-09.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <memory>

#pragma once
#ifndef CLAY_MEMORY_HPP
#define CLAY_MEMORY_HPP

namespace clay
{

std::shared_ptr<char> make_char (size_t n);

}

#endif /* CLAY_MEMORY_HPP */
