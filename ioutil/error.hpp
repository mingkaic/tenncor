/*!
 *
 *  error.hpp
 *  ioutil
 *
 *  Purpose:
 *  define commonly streamable error
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "ioutil/stream.hpp"

#pragma once
#ifndef IOUTIL_ERROR_HPP
#define IOUTIL_ERROR_HPP

namespace ioutil
{

class Error : public Stream, public std::runtime_error
{
public:
	const char* what (void) const throw ();
};

}

#endif /* IOUTIL_ERROR_HPP */
