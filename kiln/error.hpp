/*!
 *
 *  error.hpp
 *  kiln
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

#include "kiln/graph.hpp"

#pragma once
#ifndef KILN_ERROR_HPP
#define KILN_ERROR_HPP

namespace kiln
{

struct DuplicateNodeIDError : public std::runtime_error
{
	DuplicateNodeIDError (std::string gid, UID node_id);
};

struct MissingNodeError : public std::runtime_error
{
	MissingNodeError (std::string gid, UID node_id);
};

}

#endif /* KILN_ERROR_HPP */
