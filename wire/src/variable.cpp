//
//  variable.cpp
//  wire
//

#include <cassert>

#include "wire/constant.hpp"
#include "wire/variable.hpp"

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
	if (false == arg_->has_data())
	{
		throw std::exception(); // todo: add context
	}
	Identifier* out;
	clay::DTYPE otype = arg_->get_state().dtype_;
	if (this == wrt)
	{
		out = make_one(otype);
	}
	else
	{
		out = make_zero(otype);
	}
	if (nullptr == out)
	{
		throw std::exception(); // todo: add context
	}
	return out;
}

}

#endif
