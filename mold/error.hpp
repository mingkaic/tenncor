/*!
 *
 *  error.hpp
 *  mold
 *
 *  Purpose:
 *  model common node error cases
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <stdexcept>

#pragma once
#ifndef MOLD_ERROR_HPP
#define MOLD_ERROR_HPP

namespace mold
{

struct NilDataError : public std::runtime_error
{
	NilDataError (void);
};

struct UninitializedError : public std::runtime_error
{
	UninitializedError (void);
};

struct FunctorUpdateError : public std::runtime_error
{
	FunctorUpdateError (void);
};

}

#endif /* MOLD_ERROR_HPP */
