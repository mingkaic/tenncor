
#include "distrib/egrpc/client_async.hpp"
#include "distrib/egrpc/iclient.hpp"

#include "tenncor/distrib/services/io/distr.io.grpc.pb.h"

#ifndef DISTRIB_IO_CLIENT_HPP
#define DISTRIB_IO_CLIENT_HPP

namespace distr
{

struct DistrIOCli final : public egrpc::GrpcClient
{
	DistrIOCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(io::DistrInOut::NewStub(channel)),
		alias_(alias) {}

	std::future<void> list_nodes (grpc::CompletionQueue& cq,
		const io::ListNodesRequest& req,
		std::function<void(io::ListNodesResponse&)> cb)
	{
		using ListNodesHandlerT = egrpc::AsyncClientHandler<io::ListNodesResponse>;
		auto handler = new ListNodesHandlerT(
			alias_ + ":ListNodes", cb,
			[this, &req, &cq](ListNodesHandlerT* handler)
			{
				build_ctx(handler->ctx_, false);
				// prepare to avoid passing to cq before reader_ assignment
				handler->reader_ = stub_->PrepareAsyncListNodes(&handler->ctx_, req, &cq);
				// make request after reader_ assignment
				handler->reader_->StartCall();
				handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
			}, cfg_.request_retry_);
		return handler->complete_promise_.get_future();
	}

private:
	std::unique_ptr<io::DistrInOut::Stub> stub_;

	std::string alias_;
};

}

#endif // DISTRIB_IO_CLIENT_HPP
