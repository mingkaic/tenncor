
#include "egrpc/client_async.hpp"
#include "egrpc/iclient.hpp"

#include "dbg/distr_ext/print/distr.print.grpc.pb.h"

#ifndef DISTRIB_PRINT_CLIENT_HPP
#define DISTRIB_PRINT_CLIENT_HPP

namespace distr
{

struct DistrPrintCli final : public egrpc::GrpcClient
{
	DistrPrintCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(print::DistrPrint::NewStub(channel)),
		alias_(alias) {}

	egrpc::ErrFutureT list_ascii (grpc::CompletionQueue& cq,
		const print::ListAsciiRequest& req,
		std::function<void(print::AsciiEntry&)> cb)
	{
		auto handler = new egrpc::AsyncClientStreamHandler<print::AsciiEntry>(
			alias_ + ":ListAscii", cb);

		build_ctx(handler->ctx_, false);
		// prepare to avoid passing to cq before reader_ assignment
		handler->reader_ = stub_->PrepareAsyncListAscii(
			&handler->ctx_, req, &cq);
		// make request after reader_ assignment
		handler->reader_->StartCall((void*) handler);
		return handler->complete_promise_.get_future();
	}

private:
	std::unique_ptr<print::DistrPrint::Stub> stub_;

	std::string alias_;
};

}

#endif // DISTRIB_PRINT_CLIENT_HPP
