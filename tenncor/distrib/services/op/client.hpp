
#include "distrib/egrpc/client_async.hpp"
#include "distrib/egrpc/iclient.hpp"

#include "tenncor/distrib/services/op/distr.op.grpc.pb.h"

#ifndef DISTRIB_OP_CLIENT_HPP
#define DISTRIB_OP_CLIENT_HPP

namespace distr
{

struct DistrOpCli final : public egrpc::GrpcClient
{
	DistrOpCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(op::DistrOperation::NewStub(channel)),
		alias_(alias) {}

	egrpc::ErrFutureT get_data (grpc::CompletionQueue& cq,
		const op::GetDataRequest& req,
		std::function<void(op::NodeData&)> cb)
	{
		auto handler = new egrpc::AsyncClientStreamHandler<op::NodeData>(
			alias_ + ":GetData", cb);

		build_ctx(handler->ctx_, false);
		// prepare to avoid passing to cq before reader_ assignment
		handler->reader_ = stub_->PrepareAsyncGetData(
			&handler->ctx_, req, &cq);
		// make request after reader_ assignment
		handler->reader_->StartCall((void*) handler);
		return handler->complete_promise_.get_future();
	}

	std::future<void> list_reachable (grpc::CompletionQueue& cq,
		const op::ListReachableRequest& req,
		std::function<void(op::ListReachableResponse&)> cb)
	{
		using ListReachableHandlerT = egrpc::AsyncClientHandler<op::ListReachableResponse>;
		auto handler = new ListReachableHandlerT(
			alias_ + ":ListReachable", cb,
			[this, &req, &cq](ListReachableHandlerT* handler)
			{
				build_ctx(handler->ctx_, false);
				// prepare to avoid passing to cq before reader_ assignment
				handler->reader_ = stub_->PrepareAsyncListReachable(&handler->ctx_, req, &cq);
				// make request after reader_ assignment
				handler->reader_->StartCall();
				handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
			}, cfg_.request_retry_);
		return handler->complete_promise_.get_future();
	}

	std::future<void> create_derive (grpc::CompletionQueue& cq,
		const op::CreateDeriveRequest& req,
		std::function<void(op::CreateDeriveResponse&)> cb)
	{
		using CreateDeriveHandlerT = egrpc::AsyncClientHandler<op::CreateDeriveResponse>;
		auto handler = new CreateDeriveHandlerT(
			alias_ + ":Derive", cb,
			[this, &req, &cq](CreateDeriveHandlerT* handler)
			{
				build_ctx(handler->ctx_, false);
				// prepare to avoid passing to cq before reader_ assignment
				handler->reader_ = stub_->PrepareAsyncCreateDerive(&handler->ctx_, req, &cq);
				// make request after reader_ assignment
				handler->reader_->StartCall();
				handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
			}, cfg_.request_retry_);
		return handler->complete_promise_.get_future();
	}

private:
	std::unique_ptr<op::DistrOperation::Stub> stub_;

	std::string alias_;
};

using DistrOpCliPtrT = std::unique_ptr<DistrOpCli>;

}

#endif // DISTRIB_OP_CLIENT_HPP
