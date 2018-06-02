//
//  error.cpp
//  mold
//

#include "mold/error.hpp"

#ifdef MOLD_ERROR_HPP

namespace mold
{

UninitializedError::UninitializedError (void) : std::runtime_error(
	"operating on uninitialized node") {}

FunctorUpdateError::FunctorUpdateError (void) : std::runtime_error(
	"failed to update functor") {}

}

#endif
