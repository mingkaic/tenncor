#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "ade/itensor.hpp"
#include "ade/edge.hpp"

#include "ead/dbg/graph.grpc.pb.h"

#ifndef EAD_DBG_CLI_HPP
#define EAD_DBG_CLI_HPP

namespace ead
{

struct NodeMetadata
{
	bool is_func_;

	std::unordered_set<std::string> labels_;

	std::weak_ptr<ade::iTensor> ownership_;
};

using GraphCanvasT = std::unordered_map<ade::iTensor*,NodeMetadata>;

void set_transfer_request (idbg::GraphUpdateRequest& req,
	const GraphCanvasT& canvas, const ade::EdgesT& edges)
{
	auto graph = req.mutable_payload();
	assert(nullptr != graph);

	size_t id = 0;
	std::unordered_map<ade::iTensor*,size_t> id_map;
	std::vector<ade::iFunctor*> funcs;
	for (auto& node_info : canvas)
	{
		auto tens = node_info.first;
		assert(nullptr != tens);
		auto& meta = node_info.second;
		id_map.emplace(tens, id);

		idbg::Node* node = graph->add_nodes();
		assert(nullptr != node);
		node->set_id(id);
		node->set_is_func(meta.is_func_);
		google::protobuf::RepeatedPtrField<std::string> temp(
			meta.labels_.begin(), meta.labels_.end());
		node->mutable_labels()->Swap(&temp);

		if (meta.is_func_)
		{
			auto f = static_cast<ade::iFunctor*>(tens);
			funcs.push_back(f);
			node->set_repr(f->to_string() + f->shape().to_string());
		}
		else
		{
			node->set_repr(tens->to_string());
		}
		++id;
	}

	for (ade::iFunctor* f : funcs)
	{
		id = id_map[f];
		auto children = f->get_children();
		for (size_t i = 0, n = children.size(); i < n; ++i)
		{
			auto& child = children[i];
			size_t child_id = id_map[child.get_tensor().get()];

			idbg::Edge* edge = graph->add_edges();
			edge->set_parent(id);
			edge->set_child(child_id);
			edge->set_label(fmts::to_string(i));
		}
	}
	for (const ade::Edge& labelled_edge : edges)
	{
		if (false == labelled_edge.expired())
		{
			idbg::Edge* edge = graph->add_edges();
			edge->set_parent(
				id_map[labelled_edge.parent_.lock().get()]);
			edge->set_child(
				id_map[labelled_edge.child_.lock().get()]);
			edge->set_label(labelled_edge.edge_code_.name_);
		}
	}
}

struct GraphClient
{
	GraphClient (std::string host) :
		stub_(idbg::InteractiveGrapher::NewStub(
			grpc::CreateChannel(host, grpc::InsecureChannelCredentials()))) {}

	void send (const GraphCanvasT& canvas, const ade::EdgesT& edges)
	{
		grpc::ClientContext context;
		auto deadline = std::chrono::system_clock::now() +
			std::chrono::seconds(3);
		context.set_deadline(deadline);

		idbg::GraphUpdateRequest req;
		idbg::GraphUpdateResponse res;

		set_transfer_request(req, canvas, edges);

		grpc::Status status = stub_->UpdateGraph(&context, req, &res);
		if (!status.ok())
		{
			logs::error("UpdateGraph rpc failed");
			return;
		}

		auto res_status = res.status();
		if (res_status != idbg::GraphUpdateResponse_Status_OK)
		{
			logs::errorf("Not OK Response %d: %s",
				res_status, res.message().c_str());
		}
	}

	std::unique_ptr<idbg::InteractiveGrapher::Stub> stub_;
};

}

#endif // EAD_DBG_CLI_HPP
