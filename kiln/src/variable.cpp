//
//  variable.cpp
//  kiln
//

#include <cassert>

#include "kiln/variable.hpp"

#include "ioutil/stream.hpp"

#include "slip/error.hpp"

#ifdef KILN_VARIABLE_HPP

namespace kiln
{

Variable::Variable (clay::BuildTensorF builder,
	std::string label, Graph& graph) :
	Identifier(&graph, new mold::Variable(), label)
{
	graph_->uninits_[get_uid()] = builder;
}

Variable::~Variable (void)
{
	UID uid = get_uid();
	auto it = graph_->uninits_.find(uid);
	if (graph_->uninits_.end() != it)
	{
		graph_->uninits_.erase(it);
	}
}

bool Variable::assign (const Identifier& src)
{
	mold::iNode* arg = get();
	clay::State sstate = src.get_state();
	clay::State dstate = arg->get_state();
	if (dstate.dtype_ != sstate.dtype_)
	{
		throw slip::TypeMismatchError(dstate.dtype_, sstate.dtype_);
	}
	if (false == dstate.shape_.is_compatible_with(sstate.shape_))
	{
		throw slip::ShapeMismatchError(dstate.shape_, sstate.shape_);
	}
	size_t nbytes = dstate.shape_.n_elems() * clay::type_size(dstate.dtype_);
	std::memcpy(dstate.get(), sstate.get(), nbytes);
	mold::AudienceT auds = arg->get_audience();
	for (mold::iObserver* aud : auds)
	{
		aud->update();
	}
	return false;
}

}

#endif
