#include "pbm/graph.hpp"

void save_graph (tenncor::Graph& out, std::vector<ade::Tensorptr>& roots)
{
	for (ade::Tensorptr& tptr : roots)
	{
		tptr.get();
	}
	out.set_id(in.hash());
}

std::vector<ade::Tensorptr> load_graph (const tenncor::Graph& in)
{
	std::vector<ade::Tensorptr> outvec;
	return outvec;
}
