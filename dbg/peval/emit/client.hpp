///
/// client.hpp
/// emit
///
/// Purpose:
/// Implement grpc client that create and update graphs
///

#ifndef DBG_GRPC_CLIENT_HPP
#define DBG_GRPC_CLIENT_HPP

#include <chrono>

#include <grpcpp/grpcpp.h>

#include "jobs/jobs.hpp"
#include "egrpc/egrpc.hpp"

#include "dbg/peval/emit/gemitter.grpc.pb.h"

namespace emit
{

static const size_t data_sync_interval = 50;

/// GRPC client that checks for server health and make graph creation and update calls
struct GraphEmitterClient final : public egrpc::GrpcClient
{
	GraphEmitterClient (std::shared_ptr<grpc::ChannelInterface> channel,
		egrpc::ClientConfig cfg) : GrpcClient(cfg),
		stub_(gemitter::GraphEmitter::NewStub(channel)),
		connected_(true), health_checker_(
		[this]
		{
			gemitter::Empty empty;
			grpc::ClientContext context;
			build_ctx(context, true);
			gemitter::CreateModelResponse response;
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
		[this](size_t attempt, gemitter::CreateModelRequest request)
		{
			std::string sid = fmts::to_string(std::this_thread::get_id());
			if (attempt >= cfg_.request_retry_)
			{
				global::warnf("%s: CreateModelRequest max attempt exceeded", sid.c_str());
				return true;
			}
			grpc::ClientContext context;
			build_ctx(context, true);
			gemitter::CreateModelResponse response;
			grpc::Status status = this->stub_->CreateModel(
				&context, request, &response);
			if (status.ok())
			{
				auto res_status = response.status();
				if (gemitter::Status::OK != res_status)
				{
					global::errorf("%s: %s",
						gemitter::Status_Name(res_status).c_str(),
						response.message().c_str());
				}
				else
				{
					global::infof("%s: CreateModelRequest success: %s",
						sid.c_str(), response.message().c_str());
				}
				return true;
			}
			global::errorf(
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
		[this](size_t attempt,
			std::vector<gemitter::UpdateNodeDataRequest> requests,
			size_t update_it)
		{
			std::string sid = fmts::to_string(std::this_thread::get_id());
			grpc::ClientContext context;
			build_ctx(context, false);
			gemitter::UpdateNodeDataResponse response;
			std::unique_ptr<grpc::ClientWriterInterface<
				gemitter::UpdateNodeDataRequest>> writer(
				stub_->UpdateNodeData(&context, &response));

			for (auto& request : requests)
			{
				if (false == writer->Write(request))
				{
					global::errorf("failed to write update %d", update_it);
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
					global::errorf("%s: %s",
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
				global::errorf(
					"UpdateNodeData failure: %s",
					status.error_message().c_str());
			}
			global::warnf("%s: UpdateNodeData terminating", sid.c_str());
			return true;
		}, std::move(requests), std::move(update_it));
	}

	void delete_model (const std::string& eval_id)
	{
		gemitter::DeleteModelRequest request;
		request.set_model_id(eval_id);
		sequential_jobs_.attach_job(
		[this](size_t attempt, gemitter::DeleteModelRequest request)
		{
			std::string sid = fmts::to_string(std::this_thread::get_id());
			if (attempt >= cfg_.request_retry_)
			{
				global::warnf("%s: DeleteModel max attempt exceeded", sid.c_str());
				return true;
			}
			grpc::ClientContext context;
			build_ctx(context, true);
			gemitter::DeleteModelResponse response;
			grpc::Status status = this->stub_->DeleteModel(
				&context, request, &response);
			if (status.ok())
			{
				auto res_status = response.status();
				if (gemitter::Status::OK != res_status)
				{
					global::errorf("%s: %s",
						gemitter::Status_Name(res_status).c_str(),
						response.message().c_str());
				}
				else
				{
					global::infof("%s: DeleteModel success: %s",
						sid.c_str(), response.message().c_str());
				}
				return true;
			}
			global::errorf(
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

	// every request from emitter has dependency on the previous request
	jobs::Sequence sequential_jobs_;

	// connection state
	std::atomic<bool> connected_;
	jobs::ManagedJob health_checker_;
};

}

#endif // DBG_GRPC_CLIENT_HPP
