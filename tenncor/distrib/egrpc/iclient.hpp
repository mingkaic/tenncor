
#include <chrono>

#include <grpcpp/grpcpp.h>

#ifndef EGRPC_ICLIENT_HPP
#define EGRPC_ICLIENT_HPP

namespace egrpc
{

/// Configuration wrapper for creating the client
struct ClientConfig
{
	ClientConfig (void) = default;

	ClientConfig (
		std::chrono::duration<int64_t,std::milli> request_duration,
		std::chrono::duration<int64_t,std::milli> stream_duration,
		size_t request_retries) :
		request_duration_(request_duration),
		stream_duration_(stream_duration),
		request_retry_(request_retries) {}

	/// Request timeout
	std::chrono::duration<int64_t,std::milli>
	request_duration_ = std::chrono::milliseconds(250);

	/// Stream timeout
	std::chrono::duration<int64_t,std::milli>
	stream_duration_ = std::chrono::milliseconds(10000);

	size_t request_retry_ = 5;
};

struct GrpcClient
{
	GrpcClient (const ClientConfig& cfg) : cfg_(cfg) {}

	virtual ~GrpcClient (void) = default;

protected:
	void build_ctx (grpc::ClientContext& ctx, bool is_request)
	{
		// set context deadline
		std::chrono::time_point<std::chrono::system_clock> deadline =
			std::chrono::system_clock::now() +
			(is_request ? cfg_.request_duration_ : cfg_.stream_duration_);
		ctx.set_deadline(deadline);
	}

	ClientConfig cfg_;
};

}

#endif // EGRPC_ICLIENT_HPP
