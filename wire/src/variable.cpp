//
//  variable.cpp
//  wire
//

#include <cassert>

#include "wire/variable.hpp"
#include "wire/constant.hpp"

#ifdef WIRE_VARIABLE_HPP

namespace wire
{

Variable::Variable (const clay::iBuilder& builder,
	std::string label, Graph& graph) :
	Identifier(&graph, new mold::Variable(), label)
{
	graph_->uninits_[get_uid()] = std::unique_ptr<clay::iBuilder>(builder.clone());
}

Variable::Variable (const clay::iBuilder& builder,
	clay::Shape shape, std::string label, Graph& graph) :
	Identifier(&graph, new mold::Variable(), label)
{
	std::string uid = get_uid();
	graph_->alloweds_[uid] = shape;
	graph_->uninits_[uid] = std::unique_ptr<clay::iBuilder>(builder.clone());
}

Variable::~Variable (void)
{
	std::string uid = get_uid();
	auto it = graph_->uninits_.find(uid);
	if (graph_->uninits_.end() != it)
	{
		graph_->uninits_.erase(it);
	}
}

Identifier* Variable::derive (Identifier* wrt)
{
	if (false == args_[0]->has_data())
	{
		throw mold::UninitializedError();
	}
	Identifier* out;
	clay::State state = args_[0]->get_state();
	if (this == wrt)
	{
		out = make_one(state.shape_, state.dtype_);
	}
	else
	{
		out = make_zero(state.shape_, state.dtype_);
	}
	return out;
}

}

#endif
