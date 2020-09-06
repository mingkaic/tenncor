
#ifndef DISTRIB_OX_CLIENT_HPP
#define DISTRIB_OX_CLIENT_HPP

#include "egrpc/egrpc.hpp"

#include "tenncor/serial/oxsvc/distr.ox.grpc.pb.h"

namespace distr
{

namespace ox
{

struct DistrSerializeCli final : public egrpc::GrpcClient
{
	DistrSerializeCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(DistrSerialization::NewStub(channel)),
		alias_(alias) {}

	egrpc::ErrPromiseptrT get_save_graph (
		grpc::CompletionQueue& cq,
		const GetSaveGraphRequest& req,
		std::function<void(GetSaveGraphResponse&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using GetSaveGraphHandlerT = egrpc::AsyncClientHandler<GetSaveGraphResponse>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:GetSaveGraph] ", alias_.c_str()));
		new GetSaveGraphHandlerT(done, logger, cb,
			[this, &req, &cq](GetSaveGraphHandlerT* handler)
			{
				build_ctx(handler->ctx_, false);
				// prepare to avoid passing to cq before reader_ assignment
				handler->reader_ = stub_->PrepareAsyncGetSaveGraph(&handler->ctx_, req, &cq);
				// make request after reader_ assignment
				handler->reader_->StartCall();
				handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
			}, cfg_.request_retry_);
		return done;
	}

	egrpc::ErrPromiseptrT post_load_graph (
		grpc::CompletionQueue& cq,
		const PostLoadGraphRequest& req,
		std::function<void(PostLoadGraphResponse&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using PostLoadGraphHandlerT = egrpc::AsyncClientHandler<PostLoadGraphResponse>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:PostLoadGraph] ", alias_.c_str()));
		new PostLoadGraphHandlerT(done, logger, cb,
			[this, &req, &cq](PostLoadGraphHandlerT* handler)
			{
				build_ctx(handler->ctx_, false);
				// prepare to avoid passing to cq before reader_ assignment
				handler->reader_ = stub_->PrepareAsyncPostLoadGraph(&handler->ctx_, req, &cq);
				// make request after reader_ assignment
				handler->reader_->StartCall();
				handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
			}, cfg_.request_retry_);
		return done;
	}

private:
	std::unique_ptr<DistrSerialization::Stub> stub_;

	std::string alias_;
};

}

}

#endif // DISTRIB_OX_CLIENT_HPP