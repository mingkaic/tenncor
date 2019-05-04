#include "ade/ifunctor.hpp"

#ifndef ADE_OPFUNC_HPP
#define ADE_OPFUNC_HPP

namespace ade
{

/// A functor node with direct access to evaluated data
struct iOperableFunc : public ade::iFunctor
{
	virtual ~iOperableFunc (void) = default;

	virtual void update (void) = 0;

	virtual void* raw_data (void) = 0;
};

}

#endif // ADE_OPFUNC_HPP
