#include "teq/teq.hpp"

#include "distrib/async_cli.hpp"
#include "distrib/isession.hpp"

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

	grpc::Status lookup_node (
		const distr::FindNodesRequest& req, distr::FindNodesResponse& res)
	{
		grpc::ClientContext context;
		build_ctx(context, true);
		return stub_->FindNodes(&context, req, &res);
	}

	void get_data (grpc::CompletionQueue& cq, const distr::GetDataRequest& req,
		boost::bimap<std::string,teq::TensptrT>& shared_nodes)
	{
		auto handler = new AsyncHandler<distr::NodeData>(
			[&shared_nodes](distr::NodeData& res)
			{
				auto uuid = res.uuid();
				auto ref = static_cast<iDistRef*>(shared_nodes.left.at(uuid).get());
				ref->update_data(res.data().data(), res.version());
			});

		build_ctx(handler->ctx_, false);
		handler->reader_ = stub_->AsyncGetData(
			&handler->ctx_, req, &cq, (void*) handler);
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
