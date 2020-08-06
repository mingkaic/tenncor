
#include <future>

#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/async_stream.h>

#include "teq/teq.hpp"

#ifndef EGRPC_CLIENT_ASYNC_HPP
#define EGRPC_CLIENT_ASYNC_HPP

namespace egrpc
{

// Detached client response handlers
struct iClientHandler
{
	virtual ~iClientHandler  (void) = default;

	virtual void handle (bool event_status) = 0;
};

template <typename RES>
struct AsyncClientHandler final : public iClientHandler
{
	using ReadptrT = std::unique_ptr<
		grpc::ClientAsyncResponseReader<RES>>;

	using HandleResF = std::function<void(RES&)>;

	using InitF = std::function<void(AsyncClientHandler<RES>*)>;

	AsyncClientHandler (const std::string& alias,
		HandleResF cb, InitF init, size_t nretries) :
		alias_(alias), cb_(cb), init_(init), nretries_(nretries)
	{
		init_(this);
	}

	~AsyncClientHandler (void) { complete_promise_.set_value(); }

	void handle (bool event_status) override
	{
		if (status_.ok())
		{
			teq::infof("[client %s] call %p completed successfully",
				alias_.c_str(), this);
			cb_(reply_);
		}
		else
		{
			teq::errorf("[client %s] call %p (%d attempts remaining) failed: %s",
				alias_.c_str(), this, nretries_, status_.error_message().c_str());
			if (nretries_ > 0)
			{
				auto next = new AsyncClientHandler<RES>(alias_, cb_, init_, nretries_ - 1);
				next->complete_promise_.swap(complete_promise_);
			}
		}
		delete this;
	}

	std::string alias_;

	HandleResF cb_;

	InitF init_;

	size_t nretries_;

	RES reply_;

	ReadptrT reader_;

	grpc::Status status_;

	// ctx_ and reader_ need to be kept in memory
	grpc::ClientContext ctx_;

	std::promise<void> complete_promise_;
};

using ErrPromiseT = std::promise<error::ErrptrT>;

using ErrFutureT = std::future<error::ErrptrT>;

template <typename DATA>
struct AsyncClientStreamHandler final : public iClientHandler
{
	using ReadptrT = std::unique_ptr<
		grpc::ClientAsyncReaderInterface<DATA>>;

	using HandlerF = std::function<void(DATA&)>;

	AsyncClientStreamHandler (const std::string& alias, HandlerF handler) :
		alias_(alias), handler_(handler), call_status_(STARTUP) {}

	~AsyncClientStreamHandler (void) { complete_promise_.set_value(error_); }

	void handle (bool event_status) override
	{
		assert(nullptr != reader_);
		switch (call_status_)
		{
		case STARTUP:
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
				error_ = error::error(status_.error_message());
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

	ErrPromiseT complete_promise_;

	error::ErrptrT error_ = nullptr;

private:
	enum CallStatus { STARTUP, PROCESS, FINISH };

	CallStatus call_status_;
};

}

#endif // EGRPC_CLIENT_ASYNC_HPP
