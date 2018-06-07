//
//  variable.cpp
//  wire
//

#include <cassert>

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
	UID uid = get_uid();
	graph_->alloweds_[uid] = shape;
	graph_->uninits_[uid] = std::unique_ptr<clay::iBuilder>(builder.clone());
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

}

#endif
