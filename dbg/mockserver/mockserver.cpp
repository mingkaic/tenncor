#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "dbg/grpc/tenncor.grpc.pb.h"

struct GraphEmitterImpl final : public tenncor::GraphEmitter::Service
{
	grpc::Status HealthCheck (grpc::ServerContext* context,
		const tenncor::Empty* request, tenncor::Empty* response) override
	{
		return grpc::Status::OK;
	}

	// Create or update listed nodes and edges
	grpc::Status CreateGraph(grpc::ServerContext* context,
		const tenncor::CreateGraphRequest* request,
		tenncor::CreateGraphResponse* response) override
	{
		const tenncor::GraphInfo& graph = request->payload();
		const std::string& gid = graph.graph_id();
		const google::protobuf::RepeatedPtrField<
			tenncor::NodeInfo>& nodes = graph.nodes();
		const google::protobuf::RepeatedPtrField<
			tenncor::EdgeInfo>& edges = graph.edges();

		std::cout << "storing for graph " << gid << std::endl;
		if (gid_ != gid)
		{
			// reset node and edges (assume 1 client at a time)
			gid_ = gid;
			nodes_.clear();
			edges_.clear();
		}

		for (const tenncor::NodeInfo& node : nodes)
		{
			auto shape = node.shape();
			auto tags = node.tags();
			auto loc = node.location();
			nodes_.emplace(node.id(), Node{
				std::vector<uint32_t>(shape.begin(), shape.end()),
				std::unordered_map<std::string,std::string>(
					tags.begin(), tags.end()),
				loc.maxheight(),
				loc.minheight(),
			});
		}

		for (const tenncor::EdgeInfo& edge : edges)
		{
			auto parent = edge.parent();
			auto child = edge.child();

			auto pit = nodes_.find(parent);
			auto cit = nodes_.find(child);
			if (nodes_.end() == pit)
			{
				std::cerr << "for session " << gid << " parent node "
					<< parent << " didn't arrive before the edge" << std::endl;
			}
			if (nodes_.end() == cit)
			{
				std::cerr << "for session " << gid << " child node "
					<< child << " didn't arrive before the edge" << std::endl;
			}

			edges_.push_back(Edge{
				parent, child, edge.label(), edge.shaper(), edge.coorder(),
			});
		}

		response->set_status(tenncor::OK);
		response->set_message("Created Graph");
		return grpc::Status::OK;
	}

	// Update data (tensor data) of existing nodes
    grpc::Status UpdateNodeData(grpc::ServerContext* context,
		grpc::ServerReader<tenncor::UpdateNodeDataRequest>* reader,
		tenncor::UpdateNodeDataResponse* response) override
	{
		std::cout << "recording node for graph " << gid_ << std::endl;
		tenncor::UpdateNodeDataRequest req;
		while (reader->Read(&req))
		{
			auto node_data = req.payload();
			auto id = node_data.id();
			auto it = nodes_.find(id);
			if (nodes_.end() == it)
			{
				std::cerr << "for session " << gid_ << " data node "
					<< id << " didn't arrive before the data" << std::endl;
			}
		}

		response->set_status(tenncor::OK);
		response->set_message("Updated Node Data");
		return grpc::Status::OK;
	}

	struct Node
	{
		std::vector<uint32_t> shape_;

		std::unordered_map<std::string,std::string> tags_;

		uint32_t max_height_;

		uint32_t min_height_;
	};

	struct Edge
	{
		int32_t parent_id_;

		int32_t child_id_;

		std::string label_;

		std::string shaper_;

		std::string coorder_;
	};

	std::string gid_;

	std::unordered_map<int32_t,Node> nodes_;

	std::vector<Edge> edges_;
};

int main (int argc, char** argv)
{
	std::string server_address("0.0.0.0:50051");
	GraphEmitterImpl service;

	grpc::ServerBuilder builder;
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	builder.RegisterService(&service);
	std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
	std::cout << "Server listening on " << server_address << std::endl;
	server->Wait();

	return 0;
}
