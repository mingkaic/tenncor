
#include "distrib/async.hpp"
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

	size_t request_retry_ = 5;

	/// Request timeout
	std::chrono::duration<int64_t,std::milli>
	request_duration_ = std::chrono::milliseconds(250);

	/// Stream timeout
	std::chrono::duration<int64_t,std::milli>
	stream_duration_ = std::chrono::milliseconds(10000);
};

struct DistrCli final
{
	DistrCli (std::shared_ptr<grpc::Channel> channel,
		const std::string& alias, const ClientConfig& cfg) :
		stub_(distr::DistrManager::NewStub(channel)),
		alias_(alias), cfg_(cfg) {}

	grpc::Status lookup_node (
		const distr::FindNodesRequest& req, distr::FindNodesResponse& res)
	{
		grpc::ClientContext context;
		build_ctx(context, true);
		teq::infof("[client %s:FindNodes] initial call", alias_.c_str());
		auto status = stub_->FindNodes(&context, req, &res);
		for (size_t i = 1; false == status.ok() && i < cfg_.request_retry_; ++i)
		{
			teq::infof("[client %s:FindNodes] previous call failed... "
				"reattempt %d", alias_.c_str(), i);
			grpc::ClientContext context;
			build_ctx(context, true);
			status = stub_->FindNodes(&context, req, &res);
		}
		return status;
	}

	std::future<void> get_data (grpc::CompletionQueue& cq,
		const distr::GetDataRequest& req,
		boost::bimap<std::string,teq::TensptrT>& shared_nodes)
	{
		auto handler = new AsyncCliRespHandler<distr::NodeData>(alias_ + ":GetData",
			[&shared_nodes](distr::NodeData& res)
			{
				auto uuid = res.uuid();
				auto ref = static_cast<iDistRef*>(shared_nodes.left.at(uuid).get());
				ref->update_data(res.data().data(), res.version());
			});

		build_ctx(handler->ctx_, false);
		// prepare to avoid passing to cq before reader_ assignment
		handler->reader_ = stub_->PrepareAsyncGetData(
			&handler->ctx_, req, &cq);
		// make request after reader_ assignment
		handler->reader_->StartCall((void*) handler);
		return handler->complete_promise_.get_future();
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

	std::string alias_;

	ClientConfig cfg_;
};

using DistrCliPtrT = std::unique_ptr<DistrCli>;

}

#endif // DISTRIB_CLIENT_HPP
