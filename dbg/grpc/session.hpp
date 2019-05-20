#include <chrono>
#include <thread>
#include <future>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "dbg/grpc/tenncor.grpc.pb.h"

#include "ead/session.hpp"

#ifndef DBG_SESSION_HPP

namespace dbg
{

struct EdgeInfo
{
	size_t parent_;
	size_t child_;
	std::string label_;
};

struct EdgeInfoHash final
{
	size_t operator() (const EdgeInfo& edge) const
	{
		std::stringstream ss;
		ss << edge.parent_ << ","
			<< edge.child_ << ","
			<< edge.label_;
		return std::hash<std::string>()(ss.str());
	}
};

inline bool operator == (const EdgeInfo& lhs, const EdgeInfo& rhs)
{
	EdgeInfoHash hasher;
	return hasher(lhs) == hasher(rhs);
}

struct InteractiveSession final : public ead::iSession
{
	InteractiveSession (std::shared_ptr<grpc::ChannelInterface> channel,
		size_t graph_id) : stub_(tenncor::GraphEmitter::NewStub(channel)),
			graph_id_(graph_id) {}

	InteractiveSession (std::string host, size_t graph_id) :
		InteractiveSession(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials()), graph_id) {}

	~InteractiveSession (void)
	{
		// trigger all exit signals
		for (auto& rjob : retry_jobs_)
		{
			rjob.second.set_value();
		}

		// wait for termination
		for (auto& rjob : retry_jobs_)
		{
			rjob.first.join();
		}
	}

	void track (ade::iTensor* root) override
	{
		sess_.track(root);

		// setup request
		tenncor::CreateGraphRequest request;
		auto payload = request.mutable_payload();
		for (auto& statpair : sess_.stat_.graphsize_)
		{
			auto tens = statpair.first;
			size_t id = node_ids_.size();
			if (node_ids_.emplace(tens, id).second)
			{
				// add to request
				auto node = payload->add_nodes();
				node->set_id(id);
				node->set_repr(tens->to_string());
				auto s = node->shape();
				google::protobuf::RepeatedField<uint32_t> shape(
					s.begin(), s.end());
				node->mutable_shape()->Swap(&shape);
			}
		}

		auto dest_edges = payload->mutable_edges();;
		for (auto ppair : sess_.parents_)
		{
			for (ade::iOperableFunc* parent : ppair.second)
			{
				if (edges_.emplace(EdgeInfo{
						node_ids_[parent],
						node_ids_[ppair.first],
						"parent-child",
					}).second)
				{
					// add to request
					tenncor::EdgeInfo* edge = dest_edges->Add();
					edge->set_parent(node_ids_[parent]);
					edge->set_child(node_ids_[ppair.first]);
					edge->set_label("parent-child");
				}
			}
		}

		// send creation request
		grpc::ClientContext context;
		tenncor::CreateGraphResponse response;
		// set context deadline
		std::chrono::time_point deadline = std::chrono::system_clock::now() +
			std::chrono::milliseconds(100);
		context.set_deadline(deadline);

		grpc::Status status = stub_->CreateGraph(
			&context, request, &response);
		if (status.ok())
		{
			logs::infof("CreateGraphRequest success: %s",
				response.message().c_str());
		}
		else
		{
			logs::errorf("CreateGraphRequest failure: %s",
				status.error_message().c_str());
			// attempt retry
			std::promise<void> exit_signal;
			std::future<void> future_obj = exit_signal.get_future();

			std::thread thread(
				[this](std::future<void> future_obj, tenncor::CreateGraphRequest request)
				{
					size_t attempt = 1;
					while (future_obj.wait_for(std::chrono::milliseconds(1)) ==
						std::future_status::timeout)
					{
						grpc::ClientContext context;
						tenncor::CreateGraphResponse response;
						// set context deadline
						std::chrono::time_point deadline = std::chrono::system_clock::now() +
							std::chrono::milliseconds(100);
						context.set_deadline(deadline);

						grpc::Status status = this->stub_->CreateGraph(
							&context, request, &response);
						if (status.ok())
						{
							logs::infof("CreateGraphRequest success: %s",
								response.message().c_str());
							return;
						}
						logs::errorf("CreateGraphRequest attempt %d failure: %s",
							attempt, status.error_message().c_str());
						std::this_thread::sleep_for(std::chrono::milliseconds(attempt * 1000));
						++attempt;
					}
				}, std::move(future_obj), std::move(request));
			retry_jobs_.push_back(std::pair<std::thread,std::promise<void>>{
				std::move(thread), std::move(exit_signal)});
		}
	}

	void update (ead::TensSetT updated = {},
		ead::TensSetT ignores = {}) override
	{
		sess_.update(updated, ignores);
	}

	std::unique_ptr<tenncor::GraphEmitter::Stub> stub_;

	size_t graph_id_;

	ead::Session sess_;

	std::unordered_map<ade::iTensor*,size_t> node_ids_;

	std::unordered_set<EdgeInfo,EdgeInfoHash> edges_;

	std::vector<std::pair<std::thread,std::promise<void>>> retry_jobs_;
};

}

#endif // DBG_SESSION_HPP
