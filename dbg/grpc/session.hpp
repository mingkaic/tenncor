#include <grpc/grpc.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "job/scope_guard.hpp"

#include "ead/session.hpp"

#include "dbg/grpc/client.hpp"

#ifndef DBG_SESSION_HPP

namespace dbg
{

// static const size_t max_attempts = 10;

// static const size_t data_sync_interval = 50;

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

struct InteractiveSession final : public ead::iSession
{
	static boost::uuids::random_generator uuid_gen_;

	InteractiveSession (std::shared_ptr<grpc::ChannelInterface> channel) :
		client_(channel)
	{
		logs::infof("created session: %s", sess_id_.c_str());
	}

	InteractiveSession (std::string host) :
		InteractiveSession(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials())) {}

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

		client_.create_graph(request);
	}

	void update (ead::TensSetT updated = {},
		ead::TensSetT ignores = {}) override
	{
		job::ScopeGuard defer([this]() { ++this->update_it_; });

		// ignore any node data updates when
		// not connected or out of sync interval
		if (false == client_.is_connected() || 0 < update_it_ % data_sync_interval)
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

		std::vector<tenncor::UpdateNodeDataRequest> requests;
		requests.reserve(sess_.requirements_.size());
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

				// create requests (bulk of the overhead)
				tenncor::UpdateNodeDataRequest request;
				auto payload = request.mutable_payload();
				payload->set_graph_id(sess_id_);
				payload->set_node_id(node_ids_[op.first]);
				google::protobuf::RepeatedField<float> field(
					data.begin(), data.end());
				payload->mutable_data()->Swap(&field);
				requests.push_back(request);
			}
		}

		client_.update_node_data(requests, update_it_);
	}

	// join indefinitely
	void join (void)
	{
		client_.join();
	}

	// join until specified deadline, then terminate all jobs in the client
	void join_then_stop (
		const std::chrono::time_point<std::chrono::system_clock>& deadline)
	{
		std::condition_variable client_done;
		std::thread timed_killer(
		[&]()
		{
			std::mutex mtx;
			std::unique_lock<std::mutex> lck(mtx);
			client_done.wait_until(lck, deadline);
			this->client_.clear();
		});
		client_.join();
		client_done.notify_one();
		timed_killer.join();
	}

	void stop (void)
	{
		client_.clear();
	}

	std::string get_session_id (void) const
	{
		return sess_id_;
	}

	std::unique_ptr<tenncor::GraphEmitter::Stub> stub_;

	ead::Session sess_;

private:
	std::string sess_id_ = boost::uuids::to_string(
		InteractiveSession::uuid_gen_());

	std::unordered_map<ade::iTensor*,size_t> node_ids_;

	std::unordered_set<EdgeInfo,EdgeInfoHash> edges_;

	size_t update_it_ = 0;

	GraphEmitterClient client_;
};

boost::uuids::random_generator InteractiveSession::uuid_gen_;

}

#endif // DBG_SESSION_HPP
