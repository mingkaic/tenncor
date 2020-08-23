
#ifndef DBG_EMIT_HPP
#define DBG_EMIT_HPP

#include "jobs/jobs.hpp"

#include "tenncor/tenncor.hpp"

#include "dbg/peval/plugin_eval.hpp"
#include "dbg/peval/emit/client.hpp"

namespace emit
{

struct Emitter final : public dbg::iPlugin
{
	Emitter (std::shared_ptr<grpc::ChannelInterface> channel,
		egrpc::ClientConfig client_cfg = egrpc::ClientConfig()) :
		client_(channel, client_cfg)
	{
		global::infof("created evaluator: %s", eval_id_.c_str());
	}

	Emitter (std::string host,
		egrpc::ClientConfig client_cfg = egrpc::ClientConfig()) :
		Emitter(grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials()), client_cfg) {}

	void process (
		const teq::TensSetT& targets,
		const teq::TensSetT& visited) override
	{
		jobs::ScopeGuard defer([this] { ++this->update_it_; });

		// ignore any node data updates when
		// not connected or out of sync interval
		if (false == client_.is_connected() || 0 < update_it_ % data_sync_interval)
		{
			return;
		}

		if (false == update_structure(targets))
		{
			update_data(visited);
		}
	}

	/// Send create graph request
	bool update_structure (const teq::TensSetT& targets)
	{
		bool changed = false;
		onnx::TensIdT existing;
		teq::LambdaVisit vis(
			[&](teq::iLeaf& leaf)
			{
				teq::iTensor* tens = &leaf;
				bool incache = estd::has(ids_.left, tens);
				changed = changed || !incache;
				if (incache)
				{
					std::string id = ids_.left.at(tens);
					existing.insert({tens, id});
				}
			},
			[&](teq::iTraveler& trav, teq::iFunctor& func)
			{
				teq::iTensor* tens = &func;
				bool incache = estd::has(ids_.left, tens);
				changed = changed || !incache;
				if (incache)
				{
					std::string id = ids_.left.at(tens);
					existing.insert({tens, id});
				}
				auto deps = func.get_dependencies();
				teq::multi_visit(trav, deps);
			});
		teq::multi_visit(vis, targets);

		if (changed)
		{
			gemitter::CreateModelRequest request;
			auto payload = request.mutable_payload();
			payload->set_model_id(eval_id_);
			tcr::save_model(*payload->mutable_model(),
				teq::TensT(targets.begin(), targets.end()), existing);
			client_.create_model(request);

			// update ids
			onnx::TensptrIdT fullids;
			tcr::load_model(fullids, payload->model());
			for (auto fullid : fullids)
			{
				ids_.insert({fullid.left.get(), fullid.right});
			}
		}
		return changed;
	}

	void update_data (const teq::TensSetT& visited)
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
				size_t nelems = leaf->shape().n_elems();
				google::protobuf::RepeatedField<float> field;
				field.Resize(nelems, 0);
				egen::type_convert(field.mutable_data(), leaf->device().data(), dtype, nelems);

				gemitter::UpdateNodeDataRequest request;
				auto payload = request.mutable_payload();
				payload->set_model_id(eval_id_);
				payload->set_node_id(ids_.left.at(leaf));
				payload->mutable_data()->Swap(&field);
				requests.push_back(request);
			}
		}

		// ignored nodes and its dependers will never fulfill requirement
		for (auto& vis : visited)
		{
			if (auto func = dynamic_cast<teq::iFunctor*>(vis))
			{
				egen::_GENERATED_DTYPE dtype =
					(egen::_GENERATED_DTYPE) func->get_meta().type_code();
				size_t nelems = func->shape().n_elems();
				google::protobuf::RepeatedField<float> field;
				field.Resize(nelems, 0);
				egen::type_convert(field.mutable_data(),
					func->device().data(), dtype, nelems);

				// create requests (bulk of the overhead)
				gemitter::UpdateNodeDataRequest request;
				auto payload = request.mutable_payload();
				payload->set_model_id(eval_id_);
				payload->set_node_id(ids_.left.at(func));
				payload->mutable_data()->Swap(&field);
				requests.push_back(request);
			}
		}

		client_.update_node_data(requests, update_it_);
	}

	void delete_model (void)
	{
		if (sent_graph_)
		{
			client_.delete_model(eval_id_);
			eval_id_ = boost::uuids::to_string(global::get_uuidengine()());
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

	/// Return evaluation id
	std::string get_eval_id (void) const
	{
		return eval_id_;
	}

private:
	/// GRPC Client
	GraphEmitterClient client_;

	std::string eval_id_ = boost::uuids::to_string(global::get_uuidengine()());

	size_t update_it_ = 0;

	bool sent_graph_ = false;

	teq::GraphStat stat_;

	onnx::TensIdT ids_;
};

}

#endif // DBG_EMIT_HPP
