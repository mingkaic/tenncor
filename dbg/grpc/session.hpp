#include <grpc/grpc.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "jobs/scope_guard.hpp"

#include "eteq/session.hpp"

#include "tag/tag.hpp"

#include "dbg/grpc/client.hpp"

#ifndef DBG_SESSION_HPP
#define DBG_SESSION_HPP

namespace dbg
{

static const std::string tag_str_key = "name";

static const std::string tag_node_type = "node_type";

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

struct InteractiveSession final : public eteq::iSession
{
	static boost::uuids::random_generator uuid_gen_;

	InteractiveSession (std::shared_ptr<grpc::ChannelInterface> channel,
		ClientConfig client_cfg = ClientConfig(),
		tag::TagRegistry& registry = tag::get_reg()) :
		registry_(registry),
		client_(channel, client_cfg)
	{
		logs::infof("created session: %s", sess_id_.c_str());
	}

	InteractiveSession (std::string host,
		ClientConfig client_cfg = ClientConfig()) :
		InteractiveSession(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials()), client_cfg) {}

	void track (teq::TensT roots) override
	{
		sess_.track(roots);

		// setup request
		tenncor::CreateGraphRequest request;
		auto payload = request.mutable_payload();
		payload->set_graph_id(sess_id_);
		for (auto& statpair : sess_.stat_.graphsize_)
		{
			auto tens = statpair.first;
			auto& range = statpair.second;
			size_t id = node_ids_.size() + 1;
			if (false == estd::has(node_ids_, tens))
			{
				node_ids_.emplace(tens, id);
				// add to request
				auto node = payload->add_nodes();
				node->set_id(id);
				auto tags = node->mutable_tags();
				{
					tenncor::Strings tag_str;
					tag_str.add_strings(tens->to_string());
					tags->insert({tag_str_key, tag_str});
				}
				{
					tenncor::Strings type_str;
					if (0 == range.upper_)
					{
						type_str.add_strings("leaf");
					}
					else
					{
						type_str.add_strings("functor");
					}
					tags->insert({tag_node_type, type_str});
				}
				{
					auto inner_tags = registry_.get_tags(tens);
					std::map<std::string,tenncor::Strings> outer_tags;
					for (auto& itags : inner_tags)
					{
						google::protobuf::RepeatedPtrField<std::string>
						field(itags.second.begin(), itags.second.end());
						tenncor::Strings otags;
						otags.mutable_strings()->Swap(&field);
						outer_tags.emplace(itags.first, otags);
					}
					tags->insert(outer_tags.begin(), outer_tags.end());
				}
				auto s = tens->shape();
				google::protobuf::RepeatedField<uint32_t> shape(
					s.begin(), s.end());
				node->mutable_shape()->Swap(&shape);
				auto location = node->mutable_location();
				location->set_maxheight(range.upper_);
				location->set_minheight(range.lower_);
			}
		}
		for (auto& statpair : sess_.stat_.graphsize_)
		{
			auto tens = statpair.first;
			auto& range = statpair.second;
			if (range.upper_ > 0)
			{
				auto f = static_cast<teq::iFunctor*>(tens);
				auto& children = f->get_children();
				for (size_t i = 0, n = children.size(); i < n; ++i)
				{
					auto& child = children[i];
					auto child_tens = child.get_tensor().get();
					auto shaper = child.get_shaper();
					auto coorder = child.get_coorder();
					std::string label = fmts::sprintf(edge_label_fmt, i);
					EdgeInfo edgeinfo{
						node_ids_[f],
						node_ids_[child_tens],
						label,
					};
					if (false == estd::has(edges_, edgeinfo))
					{
						edges_.emplace(edgeinfo);
						// add to request
						auto edge = payload->add_edges();
						edge->set_parent(node_ids_[f]);
						edge->set_child(node_ids_[child_tens]);
						edge->set_label(label);
						if (false == teq::is_identity(shaper.get()))
						{
							edge->set_shaper(shaper->to_string());
						}
						if (false == teq::is_identity(coorder.get()))
						{
							edge->set_coorder(coorder->to_string());
						}
					}
				}
			}
		}

		client_.create_graph(request);
	}

	void update (eteq::TensSetT updated = {},
		eteq::TensSetT ignores = {}) override
	{
		jobs::ScopeGuard defer([this]() { ++this->update_it_; });

		// ignore any node data updates when
		// not connected or out of sync interval
		if (false == client_.is_connected() || 0 < update_it_ % data_sync_interval)
		{
			sess_.update(updated, ignores);
			return;
		}

		// basic copy over from session::update
		std::unordered_map<teq::iOperableFunc*,eteq::SizeT> fulfilments;
		for (teq::iTensor* unodes : updated)
		{
			auto& node_parents = sess_.parents_[unodes];
			for (auto& node_parent : node_parents)
			{
				++fulfilments[node_parent].d;
			}
		}

		std::vector<tenncor::UpdateNodeDataRequest> requests;
		requests.reserve(sess_.requirements_.size());

		for (auto& statpair : sess_.stat_.graphsize_)
		{
			if (0 == statpair.second.upper_)
			{
				auto leaf = static_cast<teq::iLeaf*>(statpair.first);
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) leaf->type_code();
				std::vector<float> data;
				size_t nelems = leaf->shape().n_elems();
				egen::type_convert(data, leaf->data(), dtype, nelems);

				tenncor::UpdateNodeDataRequest request;
				auto payload = request.mutable_payload();
				payload->set_graph_id(sess_id_);
				payload->set_node_id(node_ids_[leaf]);
				google::protobuf::RepeatedField<float> field(
					data.begin(), data.end());
				payload->mutable_data()->Swap(&field);
				requests.push_back(request);
			}
		}

		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : sess_.requirements_)
		{
			// fulfilled and not ignored
			if (fulfilments[op.first].d >= op.second &&
				false == estd::has(ignores, op.first))
			{
				op.first->update();
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) op.first->type_code();
				std::vector<float> data;
				size_t nelems = op.first->shape().n_elems();
				egen::type_convert(data, op.first->data(), dtype, nelems);
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

	void update_target (eteq::TensSetT targeted,
		eteq::TensSetT updated = {}) override
	{
		jobs::ScopeGuard defer([this]() { ++this->update_it_; });

		// ignore any node data updates when
		// not connected or out of sync interval
		if (false == client_.is_connected() || 0 < update_it_ % data_sync_interval)
		{
			sess_.update_target(targeted, updated);
			return;
		}

		// basic copy over from session::update
		teq::OnceTraveler traveler;
		for (auto& tens : targeted)
		{
			tens->accept(traveler);
		}

		std::unordered_map<teq::iOperableFunc*,eteq::SizeT> fulfilments;
		for (teq::iTensor* unodes : updated)
		{
			auto& node_parents = sess_.parents_[unodes];
			for (auto& node_parent : node_parents)
			{
				++fulfilments[node_parent].d;
			}
		}

		std::vector<tenncor::UpdateNodeDataRequest> requests;
		requests.reserve(sess_.requirements_.size());

		for (auto& statpair : sess_.stat_.graphsize_)
		{
			if (0 == statpair.second.upper_)
			{
				auto leaf = static_cast<teq::iLeaf*>(statpair.first);
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) leaf->type_code();
				std::vector<float> data;
				size_t nelems = leaf->shape().n_elems();
				egen::type_convert(data, leaf->data(), dtype, nelems);

				tenncor::UpdateNodeDataRequest request;
				auto payload = request.mutable_payload();
				payload->set_graph_id(sess_id_);
				payload->set_node_id(node_ids_[leaf]);
				google::protobuf::RepeatedField<float> field(
					data.begin(), data.end());
				payload->mutable_data()->Swap(&field);
				requests.push_back(request);
			}
		}

		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : sess_.requirements_)
		{
			// fulfilled and not ignored
			if (estd::has(traveler.visited_, op.first) &&
				fulfilments[op.first].d >= op.second)
			{
				op.first->update();
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) op.first->type_code();
				std::vector<float> data;
				size_t nelems = op.first->shape().n_elems();
				egen::type_convert(data, op.first->data(), dtype, nelems);
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

	void optimize (const opt::OptCtx& rules)
	{
		sess_.optimize(rules);

		// update graph
		node_ids_.clear();
		edges_.clear();

		// setup request
		tenncor::UpdateGraphRequest request;
		auto payload = request.mutable_payload();
		payload->set_graph_id(sess_id_);
		for (auto& statpair : sess_.stat_.graphsize_)
		{
			auto tens = statpair.first;
			auto& range = statpair.second;
			size_t id = node_ids_.size() + 1;
			if (false == estd::has(node_ids_, tens))
			{
				node_ids_.emplace(tens, id);
				// add to request
				auto node = payload->add_nodes();
				node->set_id(id);
				auto tags = node->mutable_tags();
				{
					tenncor::Strings tag_str;
					tag_str.add_strings(tens->to_string());
					tags->insert({tag_str_key, tag_str});
				}
				{
					tenncor::Strings type_str;
					if (0 == range.upper_)
					{
						type_str.add_strings("leaf");
					}
					else
					{
						type_str.add_strings("functor");
					}
					tags->insert({tag_node_type, type_str});
				}
				{
					auto inner_tags = registry_.get_tags(tens);
					std::map<std::string,tenncor::Strings> outer_tags;
					for (auto& itags : inner_tags)
					{
						google::protobuf::RepeatedPtrField<std::string>
						field(itags.second.begin(), itags.second.end());
						tenncor::Strings otags;
						otags.mutable_strings()->Swap(&field);
						outer_tags.emplace(itags.first, otags);
					}
					tags->insert(outer_tags.begin(), outer_tags.end());
				}
				auto s = tens->shape();
				google::protobuf::RepeatedField<uint32_t> shape(
					s.begin(), s.end());
				node->mutable_shape()->Swap(&shape);
				auto location = node->mutable_location();
				location->set_maxheight(range.upper_);
				location->set_minheight(range.lower_);
			}
		}
		for (auto& statpair : sess_.stat_.graphsize_)
		{
			auto tens = statpair.first;
			auto& range = statpair.second;
			if (range.upper_ > 0)
			{
				auto f = static_cast<teq::iFunctor*>(tens);
				auto& children = f->get_children();
				for (size_t i = 0, n = children.size(); i < n; ++i)
				{
					auto& child = children[i];
					auto child_tens = child.get_tensor().get();
					auto shaper = child.get_shaper();
					auto coorder = child.get_coorder();
					std::string label = fmts::sprintf(edge_label_fmt, i);
					EdgeInfo edgeinfo{
						node_ids_[f],
						node_ids_[child_tens],
						label,
					};
					if (false == estd::has(edges_, edgeinfo))
					{
						edges_.emplace(edgeinfo);
						// add to request
						auto edge = payload->add_edges();
						edge->set_parent(node_ids_[f]);
						edge->set_child(node_ids_[child_tens]);
						edge->set_label(label);
						if (false == teq::is_identity(shaper.get()))
						{
							edge->set_shaper(shaper->to_string());
						}
						if (false == teq::is_identity(coorder.get()))
						{
							edge->set_coorder(coorder->to_string());
						}
					}
				}
			}
		}

		client_.update_graph(request);
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

	eteq::Session sess_;

	tag::TagRegistry& registry_;

private:
	std::string sess_id_ = boost::uuids::to_string(
		InteractiveSession::uuid_gen_());

	std::unordered_map<teq::iTensor*,size_t> node_ids_;

	std::unordered_set<EdgeInfo,EdgeInfoHash> edges_;

	size_t update_it_ = 0;

	GraphEmitterClient client_;
};

boost::uuids::random_generator InteractiveSession::uuid_gen_;

}

#endif // DBG_SESSION_HPP
