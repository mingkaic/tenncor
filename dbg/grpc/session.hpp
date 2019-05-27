#include <chrono>
#include <thread>
#include <future>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "dbg/grpc/tenncor.grpc.pb.h"

#include "ead/session.hpp"

#ifndef DBG_SESSION_HPP

namespace dbg
{

static const size_t max_attempts = 10;

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
	static boost::uuids::random_generator uuid_gen_;

	using JobT = std::pair<std::thread,std::promise<void>>;

	InteractiveSession (std::shared_ptr<grpc::ChannelInterface> channel,
		size_t graph_id) :
		stub_(tenncor::GraphEmitter::NewStub(channel)),
		graph_id_(graph_id)
	{
		logs::infof("created session: %s", sess_id_.c_str());
	}

	InteractiveSession (std::string host, size_t graph_id) :
		InteractiveSession(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials()), graph_id) {}

	~InteractiveSession (void)
	{
		// trigger all exit signals
		job_mutex_.lock();
		for (auto& sig : kill_sigs_)
		{
			sig.second.set_value();
		}
		job_mutex_.unlock();

		// wait for termination
		join();
	}

	void track (ade::iTensor* root) override
	{
		sess_.track(root);

		// setup request
		tenncor::CreateGraphRequest request;
		auto payload = request.mutable_payload();
		payload->set_graph_id(sess_id_);
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
				auto s = tens->shape();
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
			return;
		}
		std::string request_id = boost::uuids::to_string(uuid_gen_());
		logs::errorf("%s: CreateGraphRequest failure: %s",
			request_id.c_str(), status.error_message().c_str());
		// attempt retry
		std::promise<void> exit_signal;
		std::future<void> future_obj = exit_signal.get_future();

		std::thread thread(
			[this](std::future<void> future_obj,
				tenncor::CreateGraphRequest request, std::string request_id)
			{
				auto id = std::this_thread::get_id();
				size_t attempt = 1;
				while (future_obj.wait_for(std::chrono::milliseconds(1)) ==
					std::future_status::timeout && attempt < max_attempts)
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
						logs::infof("%s: CreateGraphRequest success: %s",
							request_id.c_str(), response.message().c_str());
						this->remove_job(id);
						return;
					}
					logs::errorf(
						"%s: CreateGraphRequest attempt %d failure: %s",
						request_id.c_str(), attempt,
						status.error_message().c_str());
					std::this_thread::sleep_for(
						std::chrono::milliseconds(attempt * 1000));
					++attempt;
				}
			}, std::move(future_obj), std::move(request), request_id);

		std::lock_guard<std::mutex> lock(job_mutex_);
		retry_jobs_.push_back(std::move(thread));
		kill_sigs_.emplace(thread.get_id(), std::move(exit_signal));
	}

	void update (ead::TensSetT updated = {},
		ead::TensSetT ignores = {}) override
	{
		// basic copy over from session::update
		std::unordered_map<ade::iOperableFunc*,ead::SizeT> fulfilments;
		for (ade::iTensor* unodes : updated)
		{
			auto& node_parents = sess_.parents_[unodes];
			for (auto& node_parent : node_parents)
			{
				++fulfilments[node_parent].d;
			}
		}
		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : sess_.requirements_)
		{
			// fulfilled and not ignored
			if (fulfilments[op.first].d >= op.second &&
				ignores.end() == ignores.find(op.first))
			{
				op.first->update();
				auto& op_parents = sess_.parents_[op.first];
				for (auto& op_parent : op_parents)
				{
					++fulfilments[op_parent].d;
				}
				// // mark update
				// tenncor::UpdateNodeDataRequest request;
			}
		}
	}

	void join (void)
	{
		while (true)
		{
			{
				std::lock_guard<std::mutex>(job_mutex_);
				if (retry_jobs_.empty())
				{
					return;
				}
				auto& t = retry_jobs_.front();
				if (false == t.joinable())
				{
					retry_jobs_.pop_front();
				}
			}

			if (t.joinable())
			{
				t.join();
			}
		}
	}

	void remove_job (std::thread::id request_id)
	{
		std::lock_guard<std::mutex> lock(job_mutex_);
		auto kit = kill_sigs_.find(request_id);
		if (kill_sigs_.end() != kit)
		{
			kill_sigs_.erase(kit);
		}

		auto iter = std::find_if(retry_jobs_.begin(), retry_jobs_.end(),
			[=](std::thread &t) { return t.get_id() == request_id; });
		if (iter != retry_jobs_.end())
		{
			retry_jobs_.erase(iter);
		}
	}

	std::unique_ptr<tenncor::GraphEmitter::Stub> stub_;

	size_t graph_id_;

	ead::Session sess_;

	std::unordered_map<ade::iTensor*,size_t> node_ids_;

	std::unordered_set<EdgeInfo,EdgeInfoHash> edges_;

	std::list<std::thread> retry_jobs_;

	std::unordered_map<std::thread::id,std::promise<void>> kill_sigs_;

	std::mutex job_mutex_;

	std::string sess_id_ = boost::uuids::to_string(
		InteractiveSession::uuid_gen_());
};

boost::uuids::random_generator InteractiveSession::uuid_gen_;

}

#endif // DBG_SESSION_HPP
