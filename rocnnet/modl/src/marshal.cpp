#include "rocnnet/modl/marshal.hpp"

#ifdef MODL_MARSHAL_HPP

namespace modl
{

bool save (std::ostream& outs, ade::TensptrT source,
	iMarshaler* source_graph, iTrainingContext* tctx)
{
	pbm::GraphSaver<ead::EADSaver> saver;
	source->accept(saver);

	cortenn::Layer layer;

	// save graph from source
	cortenn::Graph* graph = layer.mutable_graph();
	saver.save(*graph, source_graph->list_bases());

	// save context if we have context
	if (nullptr != tctx)
	{
		tctx->marshal_layer(layer);
	}

	return layer.SerializeToOstream(&outs);
}

void load (std::istream& ins, iMarshaler* target,
	iTrainingContext* tctx)
{
	cortenn::Layer layer;
	layer.ParseFromIstream(&ins);

	// load graph to target
	const cortenn::Graph& graph = layer.graph();
	pbm::GraphInfo info;
	pbm::load_graph<ead::EADLoader>(info, graph);
	target->set_variables(&info.tens_);

	// load context to target
	bool has_ctx = cortenn::Layer::LAYER_CONTEXT_NOT_SET !=
		layer.layer_context_case();
	bool has_training_ctx = nullptr != tctx;
	if (has_ctx != has_training_ctx)
	{
		logs::warn("either missing training context or "
			"missing layer context from protobuf");
	}
	if (has_training_ctx)
	{
		tctx->unmarshal_layer(layer);
	}
}

}

#endif
