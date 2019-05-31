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

		response->set_status(tenncor::OK);
		response->set_message("Created Graph");
		return grpc::Status::OK;
	}

	// Update metadata (shape, labels, and representation) of existing nodes
    grpc::Status UpdateNodeMeta(grpc::ServerContext* context,
		grpc::ServerReader<tenncor::UpdateNodeMetaRequest>* reader,
		tenncor::UpdateNodeMetaResponse* response) override
	{
		tenncor::UpdateNodeMetaRequest req;
		while (reader->Read(&req))
		{
			//
		}

		response->set_status(tenncor::OK);
		response->set_message("Updated Node Metadata");
		return grpc::Status::OK;
	}

	// Update data (tensor data) of existing nodes
    grpc::Status UpdateNodeData(grpc::ServerContext* context,
		grpc::ServerReader<tenncor::UpdateNodeDataRequest>* reader,
		tenncor::UpdateNodeDataResponse* response) override
	{
		tenncor::UpdateNodeDataRequest req;
		while (reader->Read(&req))
		{
			//
		}

		response->set_status(tenncor::OK);
		response->set_message("Updated Node Data");
		return grpc::Status::OK;
	}
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
