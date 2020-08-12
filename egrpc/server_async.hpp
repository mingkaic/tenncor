
#include <future>

#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/async_stream.h>

#include "global/global.hpp"

#ifndef EGRPC_SERVER_ASYNC_HPP
#define EGRPC_SERVER_ASYNC_HPP

namespace egrpc
{

// Detached server calls
struct iServerCall
{
	virtual ~iServerCall (void) = default;

	virtual void serve (void) = 0;

	virtual void shutdown (void) = 0;
};

// Async server request response call
template <typename REQ, typename RES>
struct AsyncServerCall final : public iServerCall
{
	using RequestF = std::function<void(grpc::ServerContext*,REQ*,
		grpc::ServerAsyncResponseWriter<RES>*,grpc::CompletionQueue*,
		grpc::ServerCompletionQueue*,void*)>;

	using WriteF = std::function<grpc::Status(const REQ&,RES&)>;

	AsyncServerCall (const std::string alias,
		RequestF req_call, WriteF write_call,
		grpc::ServerCompletionQueue* cq) : alias_(alias),
		req_call_(req_call), write_call_(write_call),
		cq_(cq), responder_(&ctx_), status_(PROCESS)
	{
		req_call_(&ctx_, &req_, &responder_, cq_, cq_, (void*) this);
		global::infof("[server %s] rpc %p created", alias_.c_str(), this);
	}

	void serve (void) override
	{
		switch (status_)
		{
		case PROCESS:
		{
			new AsyncServerCall(alias_, req_call_, write_call_, cq_);
			RES reply;
			status_ = FINISH;
			global::infof("[server %s] rpc %p writing", alias_.c_str(), this);
			auto out_status = write_call_(req_, reply);
			responder_.Finish(reply, out_status, this);
		}
			break;
		case FINISH:
			global::infof("[server %s] rpc %p completed", alias_.c_str(), this);
			shutdown();
		}
	}

	void shutdown (void) override
	{
		delete this;
	}

private:
	std::string alias_;

	REQ req_;

	grpc::ServerContext ctx_;

	RequestF req_call_;

	WriteF write_call_;

	grpc::ServerCompletionQueue* cq_;

	grpc::ServerAsyncResponseWriter<RES> responder_;

	enum CallStatus { PROCESS, FINISH };

	CallStatus status_;
};

// Async server request stream call
template <typename REQ, typename RES, typename RANGE,
	typename IT = typename RANGE::iterator>
struct AsyncServerStreamCall final : public iServerCall
{
	using RequestF = std::function<void(grpc::ServerContext*,REQ*,
		grpc::ServerAsyncWriter<RES>*,grpc::CompletionQueue*,
		grpc::ServerCompletionQueue*,void*)>;

	using InitF = std::function<grpc::Status(RANGE&,const REQ&)>;

	using WriteF = std::function<bool(const REQ&,IT&,RES&)>;

	AsyncServerStreamCall (const std::string& alias,
		RequestF req_call, InitF init_call, WriteF write_call,
		grpc::ServerCompletionQueue* cq) : alias_(alias),
		req_call_(req_call), init_call_(init_call), write_call_(write_call),
		cq_(cq), responder_(&ctx_), status_(STARTUP)
	{
		req_call_(&ctx_, &req_, &responder_, cq_, cq_, (void*) this);
		global::infof("[server %s] rpc %p created", alias_.c_str(), this);
	}

	void serve (void) override
	{
		switch (status_)
		{
		case STARTUP:
		{
			new AsyncServerStreamCall(
				alias_, req_call_, init_call_, write_call_, cq_);
			global::infof("[server %s] rpc %p initializing",
				alias_.c_str(), this);
			auto out_status = init_call_(ranges_, req_);
			if (false == out_status.ok())
			{
				status_ = FINISH;
				responder_.Finish(out_status, this);
				return;
			}
			it_ = ranges_.begin();
		}
			[[fallthrough]];
		case PROCESS:
		{
			status_ = PROCESS;
			if (it_ != ranges_.end())
			{
				RES reply;
				global::infof("[server %s] rpc %p writing", alias_.c_str(), this);
				bool wrote = write_call_(req_, it_, reply);
				++it_;
				if (wrote)
				{
					responder_.Write(reply, this);
					return;
				}
			}

			if (it_ == ranges_.end())
			{
				status_ = FINISH;
				responder_.Finish(grpc::Status::OK, this);
			}
		}
			break;
		case FINISH:
			global::infof("[server %s] rpc %p completed", alias_.c_str(), this);
			shutdown();
		}
	}

	void shutdown (void) override
	{
		if (status_ == PROCESS)
		{
			responder_.Finish(grpc::Status::CANCELLED, this);
		}
		delete this;
	}

private:
	std::string alias_;

	RANGE ranges_;

	IT it_;

	REQ req_;

	grpc::ServerContext ctx_;

	RequestF req_call_;

	InitF init_call_;

	WriteF write_call_;

	grpc::ServerCompletionQueue* cq_;

	grpc::ServerAsyncWriter<RES> responder_;

	enum CallStatus { STARTUP, PROCESS, FINISH };

	CallStatus status_;
};

}

#endif // EGRPC_SERVER_ASYNC_HPP
