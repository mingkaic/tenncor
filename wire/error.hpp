/*!
 *
 *  error.hpp
 *  wire
 *
 *  Purpose:
 *  model graph error cases
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <stdexcept>
#include <string>

#pragma once
#ifndef WIRE_ERROR_HPP
#define WIRE_ERROR_HPP

namespace wire
{

struct DuplicateNodeIDError : public std::runtime_error
{
	DuplicateNodeIDError (std::string gid, std::string node_id);
};

struct MissingNodeError : public std::runtime_error
{
	MissingNodeError (std::string gid, std::string node_id);
};

}

#endif /* WIRE_ERROR_HPP */
