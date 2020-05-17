#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#ifndef DBG_EMIT_HPP
#define DBG_EMIT_HPP

#include "jobs/scope_guard.hpp"

#include "eteq/serialize.hpp"

#include "dbg/psess/emit/client.hpp"
#include "dbg/psess/plugin_sess.hpp"

namespace emit
{

struct Emitter final : public dbg::iPlugin
{
	Emitter (std::shared_ptr<grpc::ChannelInterface> channel,
		ClientConfig client_cfg = ClientConfig()) :
		client_(channel, client_cfg)
	{
		teq::infof("created session: %s", sess_id_.c_str());
	}

	Emitter (std::string host,
		ClientConfig client_cfg = ClientConfig()) :
		Emitter(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials()), client_cfg) {}

	/// UUID random generator
	static boost::uuids::random_generator uuid_gen_;

	void process (const teq::TensptrSetT& tracks, teq::FuncListT& funcs) override
	{
		jobs::ScopeGuard defer([this] { ++this->update_it_; });

		// ignore any node data updates when
		// not connected or out of sync interval
		if (false == client_.is_connected() || 0 < update_it_ % data_sync_interval)
		{
			return;
		}

		if (check_diff(tracks))
		{
			delete_model();
		}
		if (false == sent_graph_)
		{
			create_model(tracks);
		}
		update_model(funcs);
	}

	bool check_diff (const teq::TensptrSetT& tracks)
	{
		teq::GraphStat tmp_stat;
		for (auto root : tracks)
		{
			root->accept(tmp_stat);
		}
		if (tmp_stat.graphsize_.size() != stat_.graphsize_.size())
		{
			return false;
		}
		for (auto tmp : tmp_stat.graphsize_)
		{
			if (false == estd::has(stat_.graphsize_, tmp.first) || (
				stat_.graphsize_[tmp.first].upper_ != tmp.second.upper_ &&
				stat_.graphsize_[tmp.first].lower_ != tmp.second.lower_))
			{
				return false;
			}
		}
		return true;
	}

	/// Send create graph request
	void create_model (const teq::TensptrSetT& tracks)
	{
		for (auto root : tracks)
		{
			root->accept(stat_);
		}

		gemitter::CreateModelRequest request;
		auto payload = request.mutable_payload();
		payload->set_model_id(sess_id_);
		eteq::save_model(*payload->mutable_model(),
			teq::TensptrsT(tracks.begin(), tracks.end()));
		onnx::TensptrIdT fullids;
		eteq::load_model(fullids, payload->model());
		for (auto fullid : fullids)
		{
			ids_.insert({fullid.left.get(), fullid.right});
		}
		client_.create_model(request);
	}

	void update_model (const teq::FuncListT& funcs)
	{
		std::vector<gemitter::UpdateNodeDataRequest> requests;
		requests.reserve(stat_.graphsize_.size());

		for (auto& statpair : stat_.graphsize_)
		{
			if (0 == statpair.second.upper_)
			{
				auto leaf = static_cast<teq::iLeaf*>(statpair.first);
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) leaf->get_meta().type_code();
				std::vector<float> data;
				size_t nelems = leaf->shape().n_elems();
				egen::type_convert(data, leaf->device().data(), dtype, nelems);

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
		for (auto& func : funcs)
		{
			egen::_GENERATED_DTYPE dtype =
				(egen::_GENERATED_DTYPE) func->get_meta().type_code();
			std::vector<float> data;
			size_t nelems = func->shape().n_elems();
			egen::type_convert(data, func->device().data(), dtype, nelems);

			// create requests (bulk of the overhead)
			gemitter::UpdateNodeDataRequest request;
			auto payload = request.mutable_payload();
			payload->set_model_id(sess_id_);
			payload->set_node_id(ids_.left.at(func));
			google::protobuf::RepeatedField<float> field(
				data.begin(), data.end());
			payload->mutable_data()->Swap(&field);
			requests.push_back(request);
		}

		client_.update_node_data(requests, update_it_);
	}

	void delete_model (void)
	{
		stat_.graphsize_.clear();
		if (sent_graph_)
		{
			client_.delete_model(sess_id_);
			sess_id_ = boost::uuids::to_string(
				Emitter::uuid_gen_());
		}
		update_it_ = 0;
		sent_graph_ = false;
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
		[&]
		{
			std::mutex mtx;
			std::unique_lock<std::mutex> lck(mtx);
			client_done.wait_until(lck, deadline);
			client_.clear();
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

private:
	/// GRPC Client
	GraphEmitterClient client_;

	std::string sess_id_ = boost::uuids::to_string(
		Emitter::uuid_gen_());

	size_t update_it_ = 0;

	bool sent_graph_ = false;

	teq::GraphStat stat_;

	onnx::TensIdT ids_;
};

boost::uuids::random_generator Emitter::uuid_gen_;

}

#endif // DBG_EMIT_HPP
