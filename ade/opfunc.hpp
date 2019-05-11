///
/// opfunc.hpp
/// ade
///
/// Purpose:
/// Define functor nodes directly hold/manipulate data
///	This differs from Functor which should not directly manipulate data
///

#include "ade/ifunctor.hpp"

#ifndef ADE_OPFUNC_HPP
#define ADE_OPFUNC_HPP

namespace ade
{

/// A functor node with direct access to evaluated data
struct iOperableFunc : public iFunctor
{
	virtual ~iOperableFunc (void) = default;

	/// Update local data-cache using this functor's operation
	virtual void update (void) = 0;

	/// Return data-cache raw pointer
	virtual void* raw_data (void) = 0;
};

}

#endif // ADE_OPFUNC_HPP