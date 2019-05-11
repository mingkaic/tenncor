#include <unordered_map>

#include "prx/codes.hpp"

#ifdef PRX_CODES_HPP

namespace prx
{

struct EnumHash
{
	template <typename T>
	size_t operator() (T e) const
	{
		return static_cast<size_t>(e);
	}
};

static std::unordered_map<GRAPH_LAYER,std::string,EnumHash> layer2name =
{
	{ FULL_CONN, "FULL_CONN" },
	{ CONV2D, "CONV2D" },
};

static std::unordered_map<std::string,GRAPH_LAYER> name2layer =
{
	{ "FULL_CONN", FULL_CONN },
	{ "CONV2D", CONV2D },
};

static std::unordered_map<GRAPH,std::string,EnumHash> graph2name =
{
	{ MPERCEPTRON, "MPERCEPTRON" },
	{ DQUALITY, "DQUALITY" },
	{ DBELIEF, "DBELIEF" },
	{ CONV_NET, "CONV_NET" },
};

static std::unordered_map<std::string,GRAPH> name2graph =
{
	{ "MPERCEPTRON", MPERCEPTRON },
	{ "DQUALITY", DQUALITY },
	{ "DBELIEF", DBELIEF },
	{ "CONV_NET", CONV_NET },
};

std::string name_layer (GRAPH_LAYER code)
{
	auto it = layer2name.find(code);
	if (layer2name.end() == it)
	{
		return "BAD_LAYER";
	}
	return it->second;
}

GRAPH_LAYER get_layer (std::string name)
{
	auto it = name2layer.find(name);
	if (name2layer.end() == it)
	{
		return BAD_LAYER;
	}
	return it->second;
}

std::string name_graph (GRAPH code)
{
	auto it = graph2name.find(code);
	if (graph2name.end() == it)
	{
		return "BAD_GRAPH";
	}
	return it->second;
}

GRAPH get_graph (std::string name)
{
	auto it = name2graph.find(name);
	if (name2graph.end() == it)
	{
		return BAD_GRAPH;
	}
	return it->second;
}

}

#endif
