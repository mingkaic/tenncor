
#ifndef DISTR_LU_CLIENT_HPP
#define DISTR_LU_CLIENT_HPP

#include "egrpc/egrpc.hpp"

#include "internal/global/global.hpp"

#include "tenncor/find/lusvc/distr.lu.grpc.pb.h"

namespace distr
{

namespace lu
{

struct DistrLuCli final : public egrpc::GrpcClient
{
	DistrLuCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(DistrLookup::NewStub(channel)),
		alias_(alias) {}

	DistrLuCli (DistrLookup::StubInterface* stub,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(stub), alias_(alias) {}

	egrpc::ErrPromiseptrT list_nodes (egrpc::iCQueue& cq,
		const ListNodesRequest& req,
		std::function<void(ListNodesResponse&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		using ListNodesHandlerT = egrpc::AsyncClientHandler<ListNodesRequest,ListNodesResponse>;
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:ListNodes] ", alias_.c_str()));
		new ListNodesHandlerT(done, logger, cb,
		[this, &req, &cq](ListNodesRequest& inreq, ListNodesHandlerT* handler)
		{
			inreq.MergeFrom(req);
			build_ctx(handler->ctx_, false);
			// prepare to avoid passing to cq before reader_ assignment
			handler->reader_ = ListNodesHandlerT::ReadptrT(stub_->PrepareAsyncListNodes(
				&handler->ctx_, inreq, cq.get_cq()).release());
			// make request after reader_ assignment
			handler->reader_->StartCall();
			handler->reader_->Finish(&handler->reply_, &handler->status_, (void*)handler);
		}, cfg_.request_retry_);
		return done;
	}

private:
	std::unique_ptr<DistrLookup::StubInterface> stub_;

	std::string alias_;
};

using DistrLuCliPtrT = std::unique_ptr<DistrLuCli>;

}

}

#endif // DISTR_LU_CLIENT_HPP
