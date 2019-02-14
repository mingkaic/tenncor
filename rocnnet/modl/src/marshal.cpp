#include "ead/serialize.hpp"

#include "rocnnet/modl/marshal.hpp"

#ifdef MODL_MARSHAL_HPP

namespace modl
{

bool save (std::ostream& outs, ade::TensptrT source,
	iMarshaler* source_graph)
{
	pbm::GraphSaver<ead::EADSaver> saver;
	source->accept(saver);

	cortenn::Graph graph;
	saver.save(graph, source_graph->list_bases());
	return graph.SerializeToOstream(&outs);
}

void load (std::istream& ins, iMarshaler* target)
{
	cortenn::Graph graph;
	graph.ParseFromIstream(&ins);
	pbm::GraphInfo info;
	pbm::load_graph<ead::EADLoader>(info, graph);
	target->set_variables(&info.tens_);
}

}

#endif
