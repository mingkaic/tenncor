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

#include "proto/serial/data.pb.h"

#include "include/utils/utils.hpp"

namespace nnutils
{

#define TENS_TYPE tenncor::TensorT

struct unsupported_type_error : public std::exception
{
	unsupported_type_error (TENS_TYPE type) : type_(type) {}

	virtual const char* what (void) const throw()
	{
		std::stringstream msg;
		msg << "unsupported type code " << type_;
		return msg.str().c_str();
	}

private:
	TENS_TYPE type_;
};

}

#endif /* TENNCOR_ERROR_HPP */
