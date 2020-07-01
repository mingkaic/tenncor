#include <grpcpp//grpcpp.h>

#ifndef DISTRIB_ASYNC_CLI_HPP
#define DISTRIB_ASYNC_CLI_HPP

#include "experimental/distrib/distr.grpc.pb.h"

namespace distrib
{

template <typename RESPONSE>
struct iResponseHandler
{
	virtual ~iResponseHandler (void) = default;

	virtual const grpc::Status& check_status (void) const = 0;

	virtual RESPONSE& get_response (void) = 0;
};

template <typename DATA>
struct iStreamHandler
{
	virtual ~iStreamHandler (void) = default;

	virtual void handle_resp (void) = 0;

	virtual grpc::Status done (void) = 0;

	virtual DATA& get_data (void) = 0;
};

template <typename RESPONSE>
struct AsyncResponseHandler final : public iResponseHandler<RESPONSE>
{
	using ReadptrT = std::unique_ptr<
		grpc::ClientAsyncResponseReader<RESPONSE>>;

	AsyncResponseHandler (ReadptrT reader) : reader_(std::move(reader))
	{
		reader_->StartCall();
		reader_->Finish(&reply_, &status_, (void*)this);
	}

	const grpc::Status& check_status (void) const override
	{
		return status_;
	}

	RESPONSE& get_response (void) override
	{
		return reply_;
	}

	grpc::Status status_;

	// Container for the data we expect from the server.
	RESPONSE reply_;

	ReadptrT reader_;
};

template <typename DATA>
struct AsyncStreamHandler final : public iStreamHandler<DATA>
{
	using ReadptrT = std::unique_ptr<
		grpc::ClientAsyncReaderInterface<DATA>>;

	AsyncStreamHandler (ReadptrT reader) : reader_(std::move(reader))
	{
		reader_->StartCall((void*) this);
	}

	void handle_resp(void) override
	{
		reader_->Read(&reply_, (void*)this);
	}

	grpc::Status done (void) override
	{
		grpc::Status status;
		reader_->Finish(&status, (void*)this);
		return status;
	}

	DATA& get_data (void) override
	{
		return reply_;
	}

	// Container for the data we expect from the server.
	DATA reply_;

	ReadptrT reader_;
};

}

#endif // DISTRIB_ASYNC_CLI_HPP
