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
#ifndef TENSOR_MEMORY_HPP
#define TENSOR_MEMORY_HPP

namespace clay
{

std::shared_ptr<char> make_char (size_t n);

}

#endif /* TENSOR_MEMORY_HPP */
