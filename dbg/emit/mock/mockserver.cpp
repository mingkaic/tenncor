#include <grpcpp/grpcpp.h>

#include "dbg/emit/gemitter.grpc.pb.h"

struct GraphEmitterImpl final : public gemitter::GraphEmitter::Service
{
	grpc::Status HealthCheck (grpc::ServerContext* context,
		const gemitter::Empty* request, gemitter::Empty* response) override
	{
		return grpc::Status::OK;
	}

	// Create or update listed nodes and edges
	grpc::Status CreateModel(grpc::ServerContext* context,
		const gemitter::CreateModelRequest* request,
		gemitter::CreateModelResponse* response) override
	{
		const gemitter::ModelPayload& info = request->payload();
		const onnx::ModelProto& model = info.model();
		gid_ = info.model_id();

		google::protobuf::util::JsonPrintOptions options;
		auto status = google::protobuf::util::MessageToJsonString(
			model, &latest_model_, options);
		if (false == status.ok())
		{
			std::cout << "failed to serialize created graph" << std::endl;
		}

		response->set_status(gemitter::OK);
		response->set_message("Created Graph");
		return grpc::Status::OK;
	}

	// Update data (tensor data) of existing nodes
	grpc::Status UpdateNodeData(grpc::ServerContext* context,
		grpc::ServerReader<gemitter::UpdateNodeDataRequest>* reader,
		gemitter::UpdateNodeDataResponse* response) override
	{
		std::cout << "recording node for graph " << gid_ << std::endl;
		gemitter::UpdateNodeDataRequest req;
		while (reader->Read(&req))
		{
			auto node_data = req.payload();
			auto gid = node_data.model_id();
			if (gid != gid_)
			{
				std::cout << "recording node meant for graph " << gid << std::endl;
			}
			std::cout << "Updating node " << node_data.node_id() << std::endl;
		}

		response->set_status(gemitter::OK);
		response->set_message("Updated Node Data");
		return grpc::Status::OK;
	}

	// Wipe and reupdate entire graph with listed nodes and edges
	grpc::Status DeleteModel(grpc::ServerContext* context,
		const gemitter::DeleteModelRequest* request,
		gemitter::DeleteModelResponse* response) override
	{
		response->set_status(gemitter::OK);
		response->set_message("Deleted Graph");
		return grpc::Status::OK;
	}

	std::string gid_;

	std::string latest_model_;
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
