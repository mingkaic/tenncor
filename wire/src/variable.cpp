//
//  variable.cpp
//  wire
//

#include <cassert>

#include "wire/variable.hpp"

#ifdef WIRE_VARIABLE_HPP

namespace wire
{

InitF builder_bind (const clay::iBuilder& b,
	std::function<InitF(clay::iBuilder*)> bif)
{
	clay::iBuilder* builder = b.clone();
	return bif(builder);
}

Variable::Variable (const clay::iBuilder& builder, 
	std::string label, Graph& graph) :
	Identifier(&graph, new mold::Variable(), label,
	builder_bind(builder, [](clay::iBuilder* builder) 
	{
		return [builder](mold::Variable* var, SetBuilderF)
		{
			var->initialize(*builder);
			delete builder;
		};
	})) {}

Variable::Variable (const clay::iBuilder& builder,
	clay::Shape shape, std::string label, Graph& graph) :
	Identifier(&graph, new mold::Variable(), label,
	builder_bind(builder, [shape](clay::iBuilder* builder) 
	{
		return [builder, shape](mold::Variable* var, SetBuilderF)
		{
			if (shape.is_fully_defined())
			{
				var->initialize(*builder, shape);
			}
			else
			{
				var->initialize(*builder);
			}
			delete builder;
		};
	})) {}

}

#endif
