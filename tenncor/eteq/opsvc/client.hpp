
#ifndef DISTR_OP_CLIENT_HPP
#define DISTR_OP_CLIENT_HPP

#include "egrpc/egrpc.hpp"

#include "internal/global/global.hpp"

#include "tenncor/eteq/opsvc/distr.op.grpc.pb.h"

namespace distr
{

namespace op
{

struct DistrOpCli final : public egrpc::GrpcClient
{
	DistrOpCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(DistrOperation::NewStub(channel)),
		alias_(alias) {}

	DistrOpCli (DistrOperation::StubInterface* stub,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg), stub_(stub), alias_(alias) {}

	egrpc::ErrPromiseptrT get_data (
		grpc::CompletionQueue& cq,
		const GetDataRequest& req,
		std::function<void(NodeData&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using GetDataHandlerT = egrpc::AsyncClientStreamHandler<NodeData>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:GetData] ", alias_.c_str()));
		auto handler = new GetDataHandlerT(done, logger, cb);

		build_ctx(handler->ctx_, false);
		// prepare to avoid passing to cq before reader_ assignment
		handler->reader_ = stub_->PrepareAsyncGetData(
			&handler->ctx_, req, &cq);
		// make request after reader_ assignment
		handler->reader_->StartCall((void*) handler);
		return done;
	}

	egrpc::ErrPromiseptrT list_reachable (
		grpc::CompletionQueue& cq,
		const ListReachableRequest& req,
		std::function<void(ListReachableResponse&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using ListReachableHandlerT = egrpc::AsyncClientHandler<ListReachableRequest,ListReachableResponse>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:ListReachable] ", alias_.c_str()));
		new ListReachableHandlerT(done, logger, cb,
		[this, &req, &cq](ListReachableRequest& inreq, ListReachableHandlerT* handler)
		{
			inreq.MergeFrom(req);
			build_ctx(handler->ctx_, false);
			// prepare to avoid passing to cq before reader_ assignment
			handler->reader_ = stub_->PrepareAsyncListReachable(&handler->ctx_, req, &cq);
			// make request after reader_ assignment
			handler->reader_->StartCall();
			handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
		}, cfg_.request_retry_);
		return done;
	}

	egrpc::ErrPromiseptrT create_derive (
		grpc::CompletionQueue& cq,
		const CreateDeriveRequest& req,
		std::function<void(CreateDeriveResponse&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using CreateDeriveHandlerT = egrpc::AsyncClientHandler<CreateDeriveRequest,CreateDeriveResponse>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:Derive] ", alias_.c_str()));
		new CreateDeriveHandlerT(done, logger, cb,
		[this, &req, &cq](CreateDeriveRequest& inreq, CreateDeriveHandlerT* handler)
		{
			inreq.MergeFrom(req);
			build_ctx(handler->ctx_, false);
			// prepare to avoid passing to cq before reader_ assignment
			handler->reader_ = stub_->PrepareAsyncCreateDerive(&handler->ctx_, req, &cq);
			// make request after reader_ assignment
			handler->reader_->StartCall();
			handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
		}, cfg_.request_retry_);
		return done;
	}

private:
	std::unique_ptr<DistrOperation::StubInterface> stub_;

	std::string alias_;
};

using DistrOpCliPtrT = std::unique_ptr<DistrOpCli>;

}

}

#endif // DISTR_OP_CLIENT_HPP
