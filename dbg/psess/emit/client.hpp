///
/// client.hpp
/// emit
///
/// Purpose:
/// Implement grpc client that create and update graphs
///

#include <chrono>

#include <grpc/grpc.h>

#include "jobs/managed_job.hpp"
#include "jobs/sequence.hpp"

#ifndef DBG_GRPC_CLIENT_HPP
#define DBG_GRPC_CLIENT_HPP

#include "dbg/psess/emit/gemitter.grpc.pb.h"

namespace emit
{

static const size_t max_attempts = 10;

static const size_t data_sync_interval = 50;

/// Configuration wrapper for creating the client
struct ClientConfig
{
	ClientConfig (void) = default;

	ClientConfig (std::chrono::duration<int64_t,std::milli> request_duration,
		std::chrono::duration<int64_t,std::milli> stream_duration) :
		request_duration_(request_duration), stream_duration_(stream_duration) {}

	/// Request timeout
	std::chrono::duration<int64_t,std::milli> request_duration_ =
		std::chrono::milliseconds(250);

	/// Stream timeout
	std::chrono::duration<int64_t,std::milli> stream_duration_ =
		std::chrono::milliseconds(10000);
};

/// GRPC client that checks for server health and make graph creation and update calls
struct GraphEmitterClient final
{
	GraphEmitterClient (std::shared_ptr<grpc::ChannelInterface> channel,
		ClientConfig cfg) :
		stub_(gemitter::GraphEmitter::NewStub(channel)),
		cfg_(cfg),
		connected_(true),
		health_checker_(
		[this]
		{
			gemitter::Empty empty;
			grpc::ClientContext context;
			gemitter::CreateModelResponse response;
			// set context deadline
			std::chrono::time_point<std::chrono::system_clock> deadline =
				std::chrono::system_clock::now() +
				std::chrono::milliseconds(1000);
			context.set_deadline(deadline);
			grpc::Status status =
				stub_->HealthCheck(&context, empty, &empty);
			this->connected_ = status.ok();
			std::this_thread::sleep_for(
				std::chrono::milliseconds(1000));
		}) {}

	~GraphEmitterClient (void)
	{
		health_checker_.stop();
		health_checker_.join();
	}

	/// Add job that pass CreateModelRequest
	void create_model (gemitter::CreateModelRequest& request)
	{
		sequential_jobs_.attach_job(
		[this](size_t attempt, gemitter::CreateModelRequest request) -> bool
		{
			std::string sid = fmts::to_string(std::this_thread::get_id());
			if (attempt >= max_attempts)
			{
				teq::warnf("%s: CreateModelRequest max attempt exceeded", sid.c_str());
				return true;
			}
			grpc::ClientContext context;
			gemitter::CreateModelResponse response;
			// set context deadline
			std::chrono::time_point<std::chrono::system_clock> deadline =
				std::chrono::system_clock::now() + cfg_.request_duration_;
			context.set_deadline(deadline);

			grpc::Status status = this->stub_->CreateModel(
				&context, request, &response);
			if (status.ok())
			{
				auto res_status = response.status();
				if (gemitter::Status::OK != res_status)
				{
					teq::errorf("%s: %s",
						gemitter::Status_Name(res_status).c_str(),
						response.message().c_str());
				}
				else
				{
					teq::infof("%s: CreateModelRequest success: %s",
						sid.c_str(), response.message().c_str());
				}
				return true;
			}
			teq::errorf(
				"%s: CreateModelRequest attempt %d failure: %s",
				sid.c_str(), attempt, status.error_message().c_str());
			std::this_thread::sleep_for(
				std::chrono::milliseconds(attempt * 1000));
			return false;
		}, std::move(request));
	}

	/// Add job that streams UpdateNodeDataRequest
	void update_node_data (
		std::vector<gemitter::UpdateNodeDataRequest>& requests,
		size_t update_it)
	{
		sequential_jobs_.attach_job(
		[this](size_t attempt, std::vector<gemitter::UpdateNodeDataRequest> requests,
			size_t update_it) -> bool
		{
			std::string sid = fmts::to_string(std::this_thread::get_id());
			gemitter::UpdateNodeDataResponse response;
			grpc::ClientContext context;
			// set context deadline
			std::chrono::time_point<std::chrono::system_clock> deadline =
				std::chrono::system_clock::now() +
				std::chrono::milliseconds(cfg_.stream_duration_);
			context.set_deadline(deadline);
			std::unique_ptr<grpc::ClientWriterInterface<
				gemitter::UpdateNodeDataRequest>> writer(
				stub_->UpdateNodeData(&context, &response));

			for (auto& request : requests)
			{
				if (false == writer->Write(request))
				{
					teq::errorf("failed to write update %d", update_it);
					break;
				}
			}
			writer->WritesDone();

			grpc::Status status = writer->Finish();
			if (status.ok())
			{
				auto res_status = response.status();
				if (gemitter::Status::OK != res_status)
				{
					teq::errorf("%s: %s",
						gemitter::Status_Name(res_status).c_str(),
						response.message().c_str());
				}
				else
				{
					return true;
				}
			}
			else
			{
				teq::errorf(
					"UpdateNodeData failure: %s",
					status.error_message().c_str());
			}
			teq::warnf("%s: UpdateNodeData terminating", sid.c_str());
			return true;
		}, std::move(requests), std::move(update_it));
	}

	void delete_model (std::string sess_id)
	{
		gemitter::DeleteModelRequest request;
		request.set_model_id(sess_id);
		sequential_jobs_.attach_job(
		[this](size_t attempt, gemitter::DeleteModelRequest request) -> bool
		{
			std::string sid = fmts::to_string(std::this_thread::get_id());
			if (attempt >= max_attempts)
			{
				teq::warnf("%s: DeleteModel max attempt exceeded", sid.c_str());
				return true;
			}
			grpc::ClientContext context;
			gemitter::DeleteModelResponse response;
			// set context deadline
			std::chrono::time_point<std::chrono::system_clock> deadline =
				std::chrono::system_clock::now() + cfg_.request_duration_;
			context.set_deadline(deadline);

			grpc::Status status = this->stub_->DeleteModel(
				&context, request, &response);
			if (status.ok())
			{
				auto res_status = response.status();
				if (gemitter::Status::OK != res_status)
				{
					teq::errorf("%s: %s",
						gemitter::Status_Name(res_status).c_str(),
						response.message().c_str());
				}
				else
				{
					teq::infof("%s: DeleteModel success: %s",
						sid.c_str(), response.message().c_str());
				}
				return true;
			}
			teq::errorf(
				"%s: DeleteModel attempt %d failure: %s",
				sid.c_str(), attempt, status.error_message().c_str());
			std::this_thread::sleep_for(
				std::chrono::milliseconds(attempt * 1000));
			return false;
		}, std::move(request));
	}

	/// Return true if the client is connected to the server
	bool is_connected (void)
	{
		return connected_;
	}

	/// Wait until all request jobs are complete
	void join (void)
	{
		sequential_jobs_.join();
	}

	/// Kill all request jobs
	void clear (void)
	{
		sequential_jobs_.stop();
	}

private:
	std::unique_ptr<gemitter::GraphEmitter::Stub> stub_;

	ClientConfig cfg_;

	// every request from emitter has dependency on the previous request
	jobs::Sequence sequential_jobs_;

	// connection state
	std::atomic<bool> connected_;
	jobs::ManagedJob health_checker_;
};

}

#endif // DBG_GRPC_CLIENT_HPP
