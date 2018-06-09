//
//  variable.cpp
//  kiln
//

#include <cassert>

#include "kiln/variable.hpp"

#ifdef KILN_VARIABLE_HPP

namespace kiln
{

Variable::Variable (clay::BuildTensorT builder,
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

}

#endif
