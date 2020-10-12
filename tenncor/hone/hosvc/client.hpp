
#ifndef DISTR_HO_CLIENT_HPP
#define DISTR_HO_CLIENT_HPP

#include "egrpc/egrpc.hpp"

#include "internal/global/global.hpp"

#include "tenncor/hone/hosvc/distr.ho.grpc.pb.h"

namespace distr
{

namespace ho
{

struct DistrHoCli final : public egrpc::GrpcClient
{
	DistrHoCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(DistrOptimization::NewStub(channel)),
		alias_(alias) {}

	egrpc::ErrPromiseptrT put_optimize (
		grpc::CompletionQueue& cq,
		const PutOptimizeRequest& req,
		std::function<void(PutOptimizeResponse&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using PutOptimizeHandlerT = egrpc::AsyncClientHandler<PutOptimizeRequest,PutOptimizeResponse>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:PutOptimize] ", alias_.c_str()));
		new PutOptimizeHandlerT(done, logger, cb,
		[this, &req, &cq](PutOptimizeRequest& inreq, PutOptimizeHandlerT* handler)
		{
			inreq.MergeFrom(req);
			build_ctx(handler->ctx_, false);
			// prepare to avoid passing to cq before reader_ assignment
			handler->reader_ = stub_->PrepareAsyncPutOptimize(&handler->ctx_, req, &cq);
			// make request after reader_ assignment
			handler->reader_->StartCall();
			handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
		}, cfg_.request_retry_);
		return done;
	}

	// egrpc::ErrPromiseptrT put_replace ()

private:
	std::unique_ptr<DistrOptimization::Stub> stub_;

	std::string alias_;
};

using DistrHoCliPtrT = std::unique_ptr<DistrHoCli>;

}

}

#endif // DISTR_HO_CLIENT_HPP
