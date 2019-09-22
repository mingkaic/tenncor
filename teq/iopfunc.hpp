///
/// opfunc.hpp
/// teq
///
/// Purpose:
/// Define functor nodes directly hold/manipulate data
///	This differs from Functor which should not directly manipulate data
///

#include "teq/ifunctor.hpp"
#include "teq/idata.hpp"

#ifndef TEQ_OPFUNC_HPP
#define TEQ_OPFUNC_HPP

namespace teq
{

/// A functor node with direct access to evaluated data
struct iOperableFunc : public iFunctor, public iData
{
	virtual ~iOperableFunc (void) = default;

	/// Update local data-cache using this functor's operation
	virtual void update (void) = 0;
};

}

#endif // TEQ_OPFUNC_HPP
