#include <vector>

#include "ead/eigen.hpp"

#ifndef PLUGIN_PLUGIN_HPP
#define PLUGIN_PLUGIN_HPP

namespace compiler
{

template <typename T>
struct iCompiledPlugin
{
	virtual ~iCompiledPlugin (void) = default;

	virtual ead::EigenptrT<T> calculate (size_t graph_id,
		std::vector<ead::TensMapT<T>*> refs) = 0;
};

}

#endif // PLUGIN_PLUGIN_HPP
