
#ifndef DISTRIB_IO_CLIENT_HPP
#define DISTRIB_IO_CLIENT_HPP

#include "egrpc/egrpc.hpp"

#include "tenncor/distrib/iosvc/distr.io.grpc.pb.h"

namespace distr
{

namespace io
{

struct DistrIOCli final : public egrpc::GrpcClient
{
	DistrIOCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(DistrInOut::NewStub(channel)),
		alias_(alias) {}

	egrpc::ErrPromiseptrT list_nodes (
		grpc::CompletionQueue& cq,
		const ListNodesRequest& req,
		std::function<void(ListNodesResponse&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using ListNodesHandlerT = egrpc::AsyncClientHandler<ListNodesResponse>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:ListNodes] ", alias_.c_str()));
		new ListNodesHandlerT(done, logger, cb,
			[this, &req, &cq](ListNodesHandlerT* handler)
			{
				build_ctx(handler->ctx_, false);
				// prepare to avoid passing to cq before reader_ assignment
				handler->reader_ = stub_->PrepareAsyncListNodes(&handler->ctx_, req, &cq);
				// make request after reader_ assignment
				handler->reader_->StartCall();
				handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
			}, cfg_.request_retry_);
		return done;
	}

private:
	std::unique_ptr<DistrInOut::Stub> stub_;

	std::string alias_;
};

}

}

#endif // DISTRIB_IO_CLIENT_HPP
