#include "teq/teq.hpp"

#include "experimental/distrib/async_cli.hpp"

#ifndef DISTRIB_CLIENT_HPP
#define DISTRIB_CLIENT_HPP

namespace distrib
{

/// Configuration wrapper for creating the client
struct ClientConfig
{
	ClientConfig (void) = default;

	ClientConfig (
		std::chrono::duration<int64_t,std::milli> request_duration,
		std::chrono::duration<int64_t,std::milli> stream_duration) :
		request_duration_(request_duration),
		stream_duration_(stream_duration) {}

	/// Request timeout
	std::chrono::duration<int64_t,std::milli>
	request_duration_ = std::chrono::milliseconds(250);

	/// Stream timeout
	std::chrono::duration<int64_t,std::milli>
	stream_duration_ = std::chrono::milliseconds(10000);
};

struct DistrCli final
{
	DistrCli (const std::string& remote, const ClientConfig& cfg) :
		stub_(distr::DistrManager::NewStub(
			grpc::CreateChannel(remote,
				grpc::InsecureChannelCredentials()))), cfg_(cfg) {}

	iResponseHandler<distr::FindNodesResponse>*
	lookup_node (grpc::CompletionQueue& cq,
		const distr::FindNodesRequest& req)
	{
		grpc::ClientContext context;
		build_ctx(context, true);

		auto call = new AsyncResponseHandler<distr::FindNodesResponse>(
			stub_->AsyncFindNodes(&context, req, &cq));
		return call;
	}

	void get_data (grpc::CompletionQueue& cq, const distr::GetDataRequest& req)
	{
		grpc::ClientContext context;
		build_ctx(context, false);

		new AsyncStreamHandler<distr::NodeData>(
			stub_->PrepareAsyncGetData(&context, req, &cq));
	}

private:
	void build_ctx (grpc::ClientContext& ctx, bool is_request)
	{
		// set context deadline
		std::chrono::time_point<std::chrono::system_clock> deadline =
			std::chrono::system_clock::now() +
			(is_request ? cfg_.request_duration_ : cfg_.stream_duration_);
		ctx.set_deadline(deadline);
	}

	std::unique_ptr<distr::DistrManager::Stub> stub_;

	ClientConfig cfg_;
};

using DistrCliPtrT = std::unique_ptr<DistrCli>;

}

#endif // DISTRIB_CLIENT_HPP
