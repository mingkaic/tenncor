#include <grpcpp//grpcpp.h>

#ifndef DISTRIB_ASYNC_CLI_HPP
#define DISTRIB_ASYNC_CLI_HPP

#include "distrib/distr.grpc.pb.h"

namespace distrib
{

struct iResponseHandler
{
	virtual ~iResponseHandler  (void) = default;

	virtual void handle (bool event_status) = 0;
};

template <typename DATA>
struct AsyncHandler final : public iResponseHandler
{
	using ReadptrT = std::unique_ptr<
		grpc::ClientAsyncReaderInterface<DATA>>;

	using HandlerF = std::function<void(DATA&)>;

	AsyncHandler (HandlerF handler) : handler_(handler), call_status_(CREATE) {}

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
	enum CallStatus { CREATE, PROCESS, PROCESSED, FINISH };

	CallStatus call_status_;
};

}

#endif // DISTRIB_ASYNC_CLI_HPP
