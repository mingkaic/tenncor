#include <chrono>
#include <thread>
#include <future>
#include <mutex>

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

static const size_t data_sync_interval = 50;

static const std::string tag_str_key = "name";

static const std::string edge_label_fmt = "parent-child-%d";

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

template <typename ...ARGS>
struct ManagedThread
{
	ManagedThread (std::function<void(std::future<void>,ARGS...)> job,
		ARGS&&... args) :
		job_(job, exit_signal_.get_future(), std::forward<ARGS>(args)...) {}

	~ManagedThread (void)
	{
		job_.detach();
		this->stop();
	}

	ManagedThread (const ManagedThread& thr) = delete;

	ManagedThread (ManagedThread&& thr) :
		exit_signal_(std::move(thr.exit_signal_)),
		job_(std::move(thr.job_)) {}

	/// return thread id
	std::thread::id get_id (void) const
	{
		return job_.get_id();
	}

	bool joinable (void) const
	{
		return job_.joinable();
	}

	/// join if joinable
	void join (void)
	{
		if (job_.joinable())
		{
			job_.join();
		}
	}

	/// stop the job_
	void stop (void)
	{
		exit_signal_.set_value();
	}

private:
	std::promise<void> exit_signal_;

	std::thread job_;
};

struct InteractiveSession final : public ead::iSession
{
	static boost::uuids::random_generator uuid_gen_;

	InteractiveSession (std::shared_ptr<grpc::ChannelInterface> channel) :
		stub_(tenncor::GraphEmitter::NewStub(channel)),
		health_checker_(
			[this](std::future<void> future_obj)
			{
				tenncor::Empty empty;
				do
				{
					grpc::ClientContext context;
					tenncor::CreateGraphResponse response;
					// set context deadline
					std::chrono::time_point deadline = std::chrono::system_clock::now() +
						std::chrono::milliseconds(1000);
					context.set_deadline(deadline);
					grpc::Status status = stub_->HealthCheck(&context, empty, &empty);
					this->connected_ = status.ok();

					std::this_thread::sleep_for(
						std::chrono::milliseconds(1000));
				}
				while (future_obj.wait_for(std::chrono::milliseconds(1)) ==
					std::future_status::timeout);
			})
	{
		logs::infof("created session: %s", sess_id_.c_str());
	}

	InteractiveSession (std::string host) :
		InteractiveSession(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials())) {}

	~InteractiveSession (void)
	{
		// trigger all exit signals
		{
			std::lock_guard<std::mutex> guard(job_mutex_);
			for (auto& job : retry_jobs_)
			{
				job.stop();
			}
		}

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
			auto& range = statpair.second;
			size_t id = node_ids_.size();
			if (node_ids_.emplace(tens, id).second)
			{
				// add to request
				auto node = payload->add_nodes();
				node->set_id(id);
				node->mutable_tags()->insert({tag_str_key, tens->to_string()});
				auto s = tens->shape();
				google::protobuf::RepeatedField<uint32_t> shape(
					s.begin(), s.end());
				node->mutable_shape()->Swap(&shape);
				auto location = node->mutable_location();
				location->set_maxheight(range.upper_);
				location->set_minheight(range.lower_);
			}

			if (range.upper_ > 0)
			{
				auto f = static_cast<ade::iFunctor*>(tens);
				auto& children = f->get_children();
				for (size_t i = 0, n = children.size(); i < n; ++i)
				{
					auto& child = children[i];
					auto child_tens = child.get_tensor().get();
					auto shaper = child.get_shaper();
					auto coorder = child.get_coorder();
					std::string label = fmts::sprintf(edge_label_fmt, i);
					if (edges_.emplace(EdgeInfo{
						node_ids_[f],
						node_ids_[child_tens],
						label,
					}).second)
					{
						// add to request
						auto edge = payload->add_edges();
						edge->set_parent(node_ids_[f]);
						edge->set_child(node_ids_[child_tens]);
						edge->set_label(label);
						if (false == ade::is_identity(shaper.get()))
						{
							edge->set_shaper(shaper->to_string());
						}
						if (false == ade::is_identity(coorder.get()))
						{
							edge->set_coorder(coorder->to_string());
						}
					}
				}
			}
		}

		create_graph(request);
	}

	void update (ead::TensSetT updated = {},
		ead::TensSetT ignores = {}) override
	{
		if (false == connected_ && 0 == update_it_ % data_sync_interval)
		{
			sess_.update(updated, ignores);
			return;
		}

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

		// todo: make this async to avoid write overhead
		tenncor::UpdateNodeDataResponse response;
		grpc::ClientContext context;
		// set context deadline
		std::chrono::time_point deadline = std::chrono::system_clock::now() +
			std::chrono::milliseconds(500);
		context.set_deadline(deadline);
		std::unique_ptr<grpc::ClientWriterInterface<
			tenncor::UpdateNodeDataRequest>> writer(
			stub_->UpdateNodeData(&context, &response));
		bool send_request = true;

		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : sess_.requirements_)
		{
			// fulfilled and not ignored
			if (fulfilments[op.first].d >= op.second &&
				ignores.end() == ignores.find(op.first))
			{
				op.first->update();
				age::_GENERATED_DTYPE dtype =
					(age::_GENERATED_DTYPE) op.first->type_code();
				std::vector<float> data;
				size_t nelems = op.first->shape().n_elems();
				age::type_convert(data, op.first->raw_data(), dtype, nelems);
				auto& op_parents = sess_.parents_[op.first];
				for (auto& op_parent : op_parents)
				{
					++fulfilments[op_parent].d;
				}

				if (send_request)
				{
					// mark update
					tenncor::UpdateNodeDataRequest request;
					auto payload = request.mutable_payload();
					payload->set_id(node_ids_[op.first]);
					google::protobuf::RepeatedField<float> field(
						data.begin(), data.end());
					payload->mutable_data()->Swap(&field);
					payload->set_update_order(update_it_);
					if (false == writer->Write(request))
					{
						logs::errorf("failed to write update %d", update_it_);
						send_request = false;
						continue;
					}
				}
			}
		}

		writer->WritesDone();

		grpc::Status status = writer->Finish();
		if (status.ok())
		{
			auto res_status = response.status();
			if (tenncor::Status::OK != res_status)
			{
				logs::errorf("%s: %s",
					tenncor::Status_Name(res_status).c_str(),
					response.message().c_str());
			}
		}
		else
		{
			logs::errorf(
				"UpdateNodeData failure: %s",
				status.error_message().c_str());
		}
		++update_it_;
	}

	void join (void)
	{
		while (true)
		{
			ManagedThread<tenncor::CreateGraphRequest>* t = nullptr;
			{
				std::lock_guard<std::mutex> guard(job_mutex_);
				if (retry_jobs_.empty())
				{
					return;
				}
				t = &retry_jobs_.front();
				if (false == t->joinable())
				{
					retry_jobs_.pop_front();
				}
			}

			if (nullptr != t && t->joinable())
			{
				t->join();
			}
		}
	}

	std::string get_session_id (void) const
	{
		return sess_id_;
	}

	std::unique_ptr<tenncor::GraphEmitter::Stub> stub_;

	ead::Session sess_;

private:
	void create_graph (tenncor::CreateGraphRequest& request)
	{
		// create a job that retries sending creation request
		ManagedThread<tenncor::CreateGraphRequest> job(
			[this](std::future<void> future_obj,
				tenncor::CreateGraphRequest request)
			{
				auto id = std::this_thread::get_id();
				std::stringstream sid;
				sid << id;
				for (size_t attempt = 1;
					future_obj.wait_for(std::chrono::milliseconds(1)) ==
					std::future_status::timeout && attempt < max_attempts;
					++attempt)
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
						auto res_status = response.status();
						if (tenncor::Status::OK != res_status)
						{
							logs::errorf("%s: %s",
								tenncor::Status_Name(res_status).c_str(),
								response.message().c_str());
						}
						else
						{
							logs::infof("%s: CreateGraphRequest success: %s",
								sid.str().c_str(), response.message().c_str());
							std::lock_guard<std::mutex> lock(job_mutex_);
							retry_jobs_.remove_if(
								[=](ManagedThread<tenncor::CreateGraphRequest> &t)
								{ return t.get_id() == id; });
							return;
						}
					}
					else
					{
						logs::errorf(
							"%s: CreateGraphRequest attempt %d failure: %s",
							sid.str().c_str(), attempt,
							status.error_message().c_str());
					}
					std::this_thread::sleep_for(
						std::chrono::milliseconds(attempt * 1000));
				}
			}, std::move(request));

		std::lock_guard<std::mutex> lock(job_mutex_);
		retry_jobs_.push_back(std::move(job));
	}

	std::string sess_id_ = boost::uuids::to_string(
		InteractiveSession::uuid_gen_());

	std::unordered_map<ade::iTensor*,size_t> node_ids_;

	std::unordered_set<EdgeInfo,EdgeInfoHash> edges_;

	std::list<ManagedThread<tenncor::CreateGraphRequest>> retry_jobs_;

	std::mutex job_mutex_;

	size_t update_it_ = 0;

	std::atomic<bool> connected_ = true;

	ManagedThread<> health_checker_;
};

boost::uuids::random_generator InteractiveSession::uuid_gen_;

}

#endif // DBG_SESSION_HPP
