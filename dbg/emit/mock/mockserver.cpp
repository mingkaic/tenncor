#include <grpcpp/grpcpp.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>
#include <google/protobuf/util/json_util.h>

#include "dbg/emit/gemitter.grpc.pb.h"

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
		const tenncor::GraphInfo& info = request->payload();
		const std::string& gid = info.graph_id();
		const tenncor::Graph& graph = info.graph();

		google::protobuf::util::JsonPrintOptions options;
		auto status = google::protobuf::util::MessageToJsonString(
			graph, &latest_graph_, options);
		if (false == status.ok())
		{
			std::cout << "failed to serialize created graph" << std::endl;
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
			auto gid = node_data.graph_id();
			if (gid != gid_)
			{
				std::cout << "recording node meant for graph " << gid << std::endl;
			}
			auto id = node_data.node_id();
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

	// Wipe and reupdate entire graph with listed nodes and edges
	grpc::Status DeleteGraph(grpc::ServerContext* context,
		const tenncor::DeleteGraphRequest* request,
		tenncor::DeleteGraphResponse* response) override
	{
		response->set_status(tenncor::OK);
		response->set_message("Deleted Graph");
		return grpc::Status::OK;
	}

	struct Node
	{
		std::vector<uint32_t> shape_;

		std::unordered_map<std::string,
			std::vector<std::string>> tags_;

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
