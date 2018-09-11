#include "llo/node.hpp"

#include "pbm/graph.pb.h"

void save_graph (tenncor::Graph& out, std::vector<ade::Tensorptr>& roots);

std::vector<ade::Tensorptr> load_graph (const tenncor::Graph& in);
