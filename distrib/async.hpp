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

	AsyncServerCall (RequestF req_call, WriteF write_call,
		grpc::ServerCompletionQueue* cq) :
		req_call_(req_call), write_call_(write_call),
		cq_(cq), responder_(&ctx_), status_(PROCESS)
	{
		teq::infof("creating call data for new connections: %p", this);
		req_call_(&ctx_, &req_, &responder_, cq_, cq_, (void*) this);
	}

	void serve (void) override
	{
		switch (status_)
		{
		case PROCESS:
		{
			new AsyncServerCall(req_call_, write_call_, cq_);
			RES reply;
			status_ = FINISH;
			auto out_status = write_call_(req_, reply);
			responder_.Finish(reply, out_status, this);
		}
			break;
		case FINISH:
			teq::infof("rpc completed for %p", this);
			shutdown();
		}
	}

	void shutdown (void) override
	{
		delete this;
	}

private:
	enum CallStatus { PROCESS, FINISH };

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

	AsyncServerStreamCall (RequestF req_call,
		InitF init_call, WriteF write_call,
		grpc::ServerCompletionQueue* cq) : req_call_(req_call),
		init_call_(init_call), write_call_(write_call),
		cq_(cq), responder_(&ctx_), status_(STARTUP)
	{
		teq::infof("creating call data for new connections: %p", this);
		req_call_(&ctx_, &req_, &responder_, cq_, cq_, (void*) this);
	}

	void serve (void) override
	{
		switch (status_)
		{
		case STARTUP:
		{
			new AsyncServerStreamCall(req_call_, init_call_, write_call_, cq_);
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
			bool called = false;
			if (it_ != ranges_.end())
			{
				RES reply;
				if (write_call_(req_, it_, reply))
				{
					responder_.Write(reply, this);
					++it_;
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
			teq::infof("rpc completed for %p", this);
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

	AsyncCliRespHandler (HandlerF handler) : handler_(handler), call_status_(CREATE) {}

	void handle (bool event_status) override
	{
		switch (call_status_)
		{
		case CREATE:
			if (event_status)
			{
				reader_->Read(&reply_, (void*) this);
				call_status_ = PROCESS;
			}
			else
			{
				reader_->Finish(&status_, (void*)this);
				call_status_ = FINISH;
			}
			break;
		case PROCESS:
			if (event_status)
			{
				handler_(reply_);
				reader_->Read(&reply_, (void*)this);
			}
			else
			{
				reader_->Finish(&status_, (void*)this);
				call_status_ = FINISH;
			}
			break;
		case FINISH:
			if (status_.ok())
			{
				teq::infof("server response completed: %p", this);
			}
			else
			{
				teq::warnf("server response failed: %p", this);
			}
			delete this;
		}
	}

	grpc::ClientContext ctx_;

	ReadptrT reader_;

	HandlerF handler_;

	DATA reply_;

	grpc::Status status_;

private:
	enum CallStatus { CREATE, PROCESS, FINISH };

	CallStatus call_status_;
};

}

#endif // DISTRIB_ASYNC_CLI_HPP
