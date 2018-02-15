/*!
 *
 *  error.hpp
 *  cnnet
 *
 *  Purpose:
 *  define custom errors
 *
 *  Created by Mingkai Chen on 2018-01-14.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_ERROR_HPP
#define TENNCOR_ERROR_HPP

#include <iostream>
#include <exception>

#include "include/tensor/type.hpp"
#include "include/utils/utils.hpp"

namespace nnutils
{

struct type_error : public std::exception
{
	type_error (TENS_TYPE type) : type_(type) {}

	virtual const char* what (void) const throw()
	{
		return ((std::string)(nnutils::formatter() << "unsupported type code " << type_)).c_str();
	}

private:
	TENS_TYPE type_;
};

}

#endif /* TENNCOR_ERROR_HPP */
