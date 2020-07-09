
#include <future>

#include <grpcpp/grpcpp.h>

#include "teq/teq.hpp"

#include "distrib/distr.grpc.pb.h"

#ifndef DISTRIB_ASYNC_CLI_HPP
#define DISTRIB_ASYNC_CLI_HPP

namespace distrib
{

// Detached server calls
struct iServerCall
{
	virtual ~iServerCall (void) = default;

	virtual void serve (void) = 0;

	virtual void shutdown (void) = 0;
};

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
		teq::infof("[server %s] rpc %p created", alias_.c_str(), this);
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
			teq::infof("[server %s] rpc %p writing", alias_.c_str(), this);
			auto out_status = write_call_(req_, reply);
			responder_.Finish(reply, out_status, this);
		}
			break;
		case FINISH:
			teq::infof("[server %s] rpc %p completed", alias_.c_str(), this);
			shutdown();
		}
	}

	void shutdown (void) override
	{
		delete this;
	}

private:
	enum CallStatus { PROCESS, FINISH };

	std::string alias_;

	REQ req_;

	grpc::ServerContext ctx_;

	RequestF req_call_;

	WriteF write_call_;

	grpc::ServerCompletionQueue* cq_;

	grpc::ServerAsyncResponseWriter<RES> responder_;

	CallStatus status_;
};

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
		teq::infof("[server %s] rpc %p created", alias_.c_str(), this);
	}

	void serve (void) override
	{
		switch (status_)
		{
		case STARTUP:
		{
			new AsyncServerStreamCall(
				alias_, req_call_, init_call_, write_call_, cq_);
			teq::infof("[server %s] rpc %p initializing",
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
				teq::infof("[server %s] rpc %p writing", alias_.c_str(), this);
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
			teq::infof("[server %s] rpc %p completed", alias_.c_str(), this);
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
	enum CallStatus { STARTUP, PROCESS, FINISH };

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

	CallStatus status_;
};

// Detached client response handlers
struct iCliRespHandler
{
	virtual ~iCliRespHandler  (void) = default;

	virtual void handle (bool event_status) = 0;
};

template <typename DATA>
struct AsyncCliRespHandler final : public iCliRespHandler
{
	using ReadptrT = std::unique_ptr<
		grpc::ClientAsyncReaderInterface<DATA>>;

	using HandlerF = std::function<void(DATA&)>;

	AsyncCliRespHandler (const std::string& alias, HandlerF handler) :
		alias_(alias), handler_(handler), call_status_(CREATE) {}

	~AsyncCliRespHandler (void) { complete_promise_.set_value(); }

	void handle (bool event_status) override
	{
		assert(nullptr != reader_);
		switch (call_status_)
		{
		case CREATE:
			if (event_status)
			{
				teq::infof("[client %s] call %p created... processing",
					alias_.c_str(), this);
				call_status_ = PROCESS;
				reader_->Read(&reply_, (void*) this);
			}
			else
			{
				teq::infof("[client %s] call %p created... finishing",
					alias_.c_str(), this);
				call_status_ = FINISH;
				reader_->Finish(&status_, (void*)this);
			}
			break;
		case PROCESS:
			if (event_status)
			{
				teq::infof("[client %s] call %p received... handling",
					alias_.c_str(), this);
				handler_(reply_);
				reader_->Read(&reply_, (void*)this);
			}
			else
			{
				teq::infof("[client %s] call %p received... finishing",
					alias_.c_str(), this);
				call_status_ = FINISH;
				reader_->Finish(&status_, (void*)this);
			}
			break;
		case FINISH:
			if (status_.ok())
			{
				teq::infof("[client %s] call %p completed successfully",
					alias_.c_str(), this);
			}
			else
			{
				teq::errorf("[client %s] call %p failed: %s",
					alias_.c_str(), this, status_.error_message().c_str());
			}
			delete this;
		}
	}

	std::string alias_;

	grpc::ClientContext ctx_;

	ReadptrT reader_;

	HandlerF handler_;

	DATA reply_;

	grpc::Status status_;

	std::promise<void> complete_promise_;

private:
	enum CallStatus { CREATE, PROCESS, FINISH };

	CallStatus call_status_;
};

}

#endif // DISTRIB_ASYNC_CLI_HPP
