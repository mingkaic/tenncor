///
/// session.hpp
/// dbg
///
/// Purpose:
/// Implement session that runs functor updates and
/// pass graph updates to GRPC server
///

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "jobs/scope_guard.hpp"

#include "teq/session.hpp"

#include "eteq/serialize.hpp"

#include "dbg/emit/client.hpp"

#ifndef DBG_SESSION_HPP
#define DBG_SESSION_HPP

namespace dbg
{

static const std::string tag_str_key = "name";

static const std::string tag_node_type = "node_type";

static const std::string edge_label_fmt = "parent-child-%d";

/// Graph edge intermediate representation
struct EdgeInfo
{
	size_t parent_;
	size_t child_;
	std::string label_;
};

/// Graph edge hashing
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

/// Graph edge equality
inline bool operator == (const EdgeInfo& lhs, const EdgeInfo& rhs)
{
	EdgeInfoHash hasher;
	return hasher(lhs) == hasher(rhs);
}

/// Session that makes GRPC client calls
struct InteractiveSession final : public teq::iSession
{
	/// UUID random generator
	static boost::uuids::random_generator uuid_gen_;

	InteractiveSession (std::shared_ptr<grpc::ChannelInterface> channel,
		ClientConfig client_cfg = ClientConfig()) :
		client_(channel, client_cfg)
	{
		logs::infof("created session: %s", sess_id_.c_str());
	}

	InteractiveSession (std::string host,
		ClientConfig client_cfg = ClientConfig()) :
		InteractiveSession(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials()), client_cfg) {}

	/// Implementation of iSession
	void track (teq::TensptrsT roots) override
	{
		sess_.track(roots);

		for (auto root : roots)
		{
			root->accept(stat_);
		}
	}

	/// Implementation of iSession
	void update (teq::TensSetT ignored = {}) override
	{
		if (false == sent_graph_)
		{
			create_model();
		}

		jobs::ScopeGuard defer([this]() { ++this->update_it_; });

		// ignore any node data updates when
		// not connected or out of sync interval
		if (false == client_.is_connected() || 0 < update_it_ % data_sync_interval)
		{
			sess_.update(ignored);
			return;
		}

		// basic copy over from session::update
		std::list<teq::iFunctor*> reqs;
		teq::TensSetT acceptable;
		for (auto& root : sess_.tracked_)
		{
			acceptable.emplace(root.get());
		}
		// ignored tensors will never populate reqs
		for (auto rit = sess_.ops_.rbegin(),
			ret = sess_.ops_.rend();
			rit != ret; ++rit)
		{
			auto& op = *rit;
			if (estd::has(acceptable, op) &&
				false == estd::has(ignored, op))
			{
				reqs.push_front(op);
				auto children = op->get_children();
				for (teq::TensptrT child : children)
				{
					acceptable.emplace(child.get());
				}
			}
		}

		std::vector<gemitter::UpdateNodeDataRequest> requests;
		requests.reserve(sess_.ops_.size());

		for (auto& statpair : stat_.graphsize_)
		{
			if (0 == statpair.second.upper_)
			{
				auto leaf = static_cast<teq::iLeaf*>(statpair.first);
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) leaf->type_code();
				std::vector<float> data;
				size_t nelems = leaf->shape().n_elems();
				egen::type_convert(data, leaf->data(), dtype, nelems);

				gemitter::UpdateNodeDataRequest request;
				auto payload = request.mutable_payload();
				payload->set_model_id(sess_id_);
				payload->set_node_id(ids_.left.at(leaf));
				google::protobuf::RepeatedField<float> field(
					data.begin(), data.end());
				payload->mutable_data()->Swap(&field);
				requests.push_back(request);
			}
		}

		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : reqs)
		{
			op->calc();
			egen::_GENERATED_DTYPE dtype =
				(egen::_GENERATED_DTYPE) op->type_code();
			std::vector<float> data;
			size_t nelems = op->shape().n_elems();
			egen::type_convert(data, op->data(), dtype, nelems);

			// create requests (bulk of the overhead)
			gemitter::UpdateNodeDataRequest request;
			auto payload = request.mutable_payload();
			payload->set_model_id(sess_id_);
			payload->set_node_id(ids_.left.at(op));
			google::protobuf::RepeatedField<float> field(
				data.begin(), data.end());
			payload->mutable_data()->Swap(&field);
			requests.push_back(request);
		}

		client_.update_node_data(requests, update_it_);
	}

	/// Implementation of iSession
	void update_target (
		teq::TensSetT targeted,
		teq::TensSetT ignored = {}) override
	{
		if (false == sent_graph_)
		{
			create_model();
		}

		jobs::ScopeGuard defer([this]() { ++this->update_it_; });

		// ignore any node data updates when
		// not connected or out of sync interval
		if (false == client_.is_connected() || 0 < update_it_ % data_sync_interval)
		{
			sess_.update_target(targeted, ignored);
			return;
		}

		// basic copy over from session::update_target
		std::list<teq::iFunctor*> reqs;
		teq::TensSetT acceptable;
		for (auto& root : targeted)
		{
			acceptable.emplace(root);
		}
		// ignored tensors will never populate reqs
		for (auto rit = sess_.ops_.rbegin(), ret = sess_.ops_.rend();
			rit != ret; ++rit)
		{
			auto& op = *rit;
			if (estd::has(acceptable, op) &&
				false == estd::has(ignored, op))
			{
				reqs.push_front(op);
				auto children = op->get_children();
				for (teq::TensptrT child : children)
				{
					acceptable.emplace(child.get());
				}
			}
		}

		std::vector<gemitter::UpdateNodeDataRequest> requests;
		requests.reserve(reqs.size());

		for (auto& statpair : stat_.graphsize_)
		{
			if (0 == statpair.second.upper_)
			{
				auto leaf = static_cast<teq::iLeaf*>(statpair.first);
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) leaf->type_code();
				std::vector<float> data;
				size_t nelems = leaf->shape().n_elems();
				egen::type_convert(data, leaf->data(), dtype, nelems);

				gemitter::UpdateNodeDataRequest request;
				auto payload = request.mutable_payload();
				payload->set_model_id(sess_id_);
				payload->set_node_id(ids_.left.at(leaf));
				google::protobuf::RepeatedField<float> field(
					data.begin(), data.end());
				payload->mutable_data()->Swap(&field);
				requests.push_back(request);
			}
		}

		// ignored nodes and its dependers will never fulfill requirement
		for (auto& op : reqs)
		{
			op->calc();
			egen::_GENERATED_DTYPE dtype =
				(egen::_GENERATED_DTYPE) op->type_code();
			std::vector<float> data;
			size_t nelems = op->shape().n_elems();
			egen::type_convert(data, op->data(), dtype, nelems);

			// create requests (bulk of the overhead)
			gemitter::UpdateNodeDataRequest request;
			auto payload = request.mutable_payload();
			payload->set_model_id(sess_id_);
			payload->set_node_id(ids_.left.at(op));
			google::protobuf::RepeatedField<float> field(
				data.begin(), data.end());
			payload->mutable_data()->Swap(&field);
			requests.push_back(request);
		}

		client_.update_node_data(requests, update_it_);
	}

	/// Implementation of iSession
	teq::TensptrSetT get_tracked (void) const override
	{
		return sess_.tracked_;
	}

	/// Implementation of iSession
	void clear (void) override
	{
		sess_.clear();
		stat_.graphsize_.clear();
		if (sent_graph_)
		{
			client_.delete_model(sess_id_);
			sess_id_ = boost::uuids::to_string(
				InteractiveSession::uuid_gen_());
		}
		update_it_ = 0;
		sent_graph_ = false;
	}

	/// Send create graph request
	void create_model (void)
	{
		gemitter::CreateModelRequest request;
		auto payload = request.mutable_payload();
		payload->set_model_id(sess_id_);
		teq::TensptrsT tracked(sess_.tracked_.begin(), sess_.tracked_.end());
		eteq::save_model(*payload->mutable_model(), tracked, ids_);
		client_.create_model(request);
	}

	/// Wait until client completes its request calls
	void join (void)
	{
		client_.join();
	}

	/// Wait until specified deadline, then terminate all jobs in the client
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

	/// Kill all request jobs
	void stop (void)
	{
		client_.clear();
	}

	/// Return session id
	std::string get_session_id (void) const
	{
		return sess_id_;
	}

	/// GRPC Client
	std::unique_ptr<gemitter::GraphEmitter::Stub> stub_;

	/// Session underneath
	teq::Session sess_;

private:
	std::string sess_id_ = boost::uuids::to_string(
		InteractiveSession::uuid_gen_());

	GraphEmitterClient client_;

	size_t update_it_ = 0;

	bool sent_graph_ = false;

	teq::GraphStat stat_;

	onnx::TensIdT ids_;
};

boost::uuids::random_generator InteractiveSession::uuid_gen_;

}

#endif // DBG_SESSION_HPP
