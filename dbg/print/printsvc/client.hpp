
#ifndef DISTRIB_PRINT_CLIENT_HPP
#define DISTRIB_PRINT_CLIENT_HPP

#include "egrpc/egrpc.hpp"

#include "dbg/print/printsvc/distr.print.grpc.pb.h"

namespace distr
{

namespace print
{

struct DistrPrintCli final : public egrpc::GrpcClient
{
	DistrPrintCli (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) :
		GrpcClient(cfg),
		stub_(DistrPrint::NewStub(channel)),
		alias_(alias) {}

	egrpc::ErrPromiseptrT list_ascii (
		grpc::CompletionQueue& cq,
		const ListAsciiRequest& req,
		std::function<void(AsciiEntry&)> cb)
	{
		auto done = std::make_shared<egrpc::ErrPromiseT>();
		auto logger = std::make_shared<global::FormatLogger>(global::get_logger(),
			fmts::sprintf("[client %s:ListAscii] ", alias_.c_str()));
		auto handler = new egrpc::AsyncClientStreamHandler<AsciiEntry>(done, logger, cb);

		build_ctx(handler->ctx_, false);
		// prepare to avoid passing to cq before reader_ assignment
		handler->reader_ = stub_->PrepareAsyncListAscii(
			&handler->ctx_, req, &cq);
		// make request after reader_ assignment
		handler->reader_->StartCall((void*) handler);
		return done;
	}

private:
	std::unique_ptr<DistrPrint::Stub> stub_;

	std::string alias_;
};

}

}

#endif // DISTRIB_PRINT_CLIENT_HPP
