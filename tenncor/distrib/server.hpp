
#include <condition_variable>
#include <mutex>
#include <thread>

#include "distrib/async.hpp"
#include "distrib/evaluator.hpp"

#ifndef DISTRIB_SERVER_HPP
#define DISTRIB_SERVER_HPP

namespace distr
{

#define ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	teq::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

struct DistrServer final
{
	using AsyncService = DistrManager::AsyncService;

	// non-blocking
	DistrServer (iDistManager* mgr,
		size_t port, std::string alias) :
		alias_(alias)
	{
		std::string address = fmts::sprintf("0.0.0.0:%d", port);
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		builder.RegisterService(&service_);
		cq_ = builder.AddCompletionQueue();
		server_ = builder.BuildAndStart();

		teq::infof("[server %s] listening on %s", alias_.c_str(), address.c_str());

		std::thread rpc_job(&DistrServer::handle_rpcs, this, mgr);
		rpc_job.detach();
	}

	~DistrServer (void)
	{
		server_->Shutdown();
		cq_->Shutdown();
		std::unique_lock<std::mutex> lk(rpc_exit_mutex_);
		rpc_exit_.wait(lk, [this]{ return bool(this->rpc_done_); });
	}

private:
	using DataStatesT = std::unordered_map<std::string,teq::iTensor*>;

	// This can be run in multiple threads if needed.
	void handle_rpcs (iDistManager* mgr)
	{
		// FindNodes
		new AsyncServerCall<FindNodesRequest,FindNodesResponse>(
		fmts::sprintf("%s:FindNodes", alias_.c_str()),
		[this](grpc::ServerContext* ctx, FindNodesRequest* req,
			grpc::ServerAsyncResponseWriter<FindNodesResponse>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_.RequestFindNodes(ctx, req, writer, cq, ccq, tag);
		},
		[this, mgr](
			const FindNodesRequest& req,
			FindNodesResponse& res)
		{
			return this->find_nodes(mgr, req, res);
		}, cq_.get());

		// FindReachable
		new AsyncServerCall<FindReachableRequest,FindReachableResponse>(
		fmts::sprintf("%s:FindReachable", alias_.c_str()),
		[this](grpc::ServerContext* ctx, FindReachableRequest* req,
			grpc::ServerAsyncResponseWriter<FindReachableResponse>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_.RequestFindReachable(ctx, req, writer, cq, ccq, tag);
		},
		[this, mgr](
			const FindReachableRequest& req,
			FindReachableResponse& res)
		{
			return this->find_reachable(mgr, req, res);
		}, cq_.get());

		// Derive
		new AsyncServerCall<DeriveRequest,DeriveResponse>(
		fmts::sprintf("%s:Derive", alias_.c_str()),
		[this](grpc::ServerContext* ctx, DeriveRequest* req,
			grpc::ServerAsyncResponseWriter<DeriveResponse>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_.RequestDerive(ctx, req, writer, cq, ccq, tag);
		},
		[this, mgr](
			const DeriveRequest& req,
			DeriveResponse& res)
		{
			return this->derive(mgr, req, res);
		}, cq_.get());

		// GetData
		new AsyncServerStreamCall<
			GetDataRequest,NodeData,DataStatesT>(
		fmts::sprintf("%s:GetData", alias_.c_str()),
		[this](grpc::ServerContext* ctx, GetDataRequest* req,
			grpc::ServerAsyncWriter<NodeData>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_.RequestGetData(ctx, req, writer, cq, ccq, tag);
		},
		[this, mgr](DataStatesT& states, const GetDataRequest& req) -> grpc::Status
		{
			auto& uuids = req.uuids();
			teq::TensSetT targets;
			for (const std::string& uuid : uuids)
			{
				error::ErrptrT err = nullptr;
				auto tens = mgr->lookup_node(err, uuid, false).get();
				ERR_CHECK(err, grpc::NOT_FOUND, this->alias_.c_str());
				targets.emplace(tens);
				states.emplace(uuid, tens);
			}
			eigen::Device device(std::numeric_limits<size_t>::max());
			DistEvaluator(mgr).evaluate(device, targets);
			return grpc::Status::OK;
		},
		[](const GetDataRequest& req, DataStatesT::iterator& it,
			NodeData& reply)
		{
			auto id = it->first;
			auto tens = it->second;

			auto& meta = tens->get_meta();
			size_t latest = meta.state_version();

			reply.set_uuid(id);
			reply.set_version(latest);

			void* raw = tens->device().data();
			auto dtype = (egen::_GENERATED_DTYPE) meta.type_code();

			size_t nelems = tens->shape().n_elems();
			google::protobuf::RepeatedField<double> field;
			field.Resize(nelems, 0);
			egen::type_convert(field.mutable_data(), raw, dtype, nelems);
			reply.mutable_data()->Swap(&field);
			return true;
		}, cq_.get());

		void* tag;
		bool ok = true;
		while (cq_->Next(&tag, &ok))
		{
			auto call = static_cast<iServerCall*>(tag);
			if (ok)
			{
				call->serve();
			}
			else
			{
				call->shutdown();
			}
		}
		rpc_done_ = true;
		rpc_exit_.notify_one();
	}

	grpc::Status find_nodes (
		iDistManager* mgr,
		const FindNodesRequest& req,
		FindNodesResponse& res)
	{
		auto& uuids = req.uuids();
		for (const std::string& uuid : uuids)
		{
			error::ErrptrT err = nullptr;
			auto node = mgr->lookup_node(err, uuid, false);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());

			NodeMeta* out = res.add_values();
			out->set_uuid(uuid);
			auto& meta = node->get_meta();
			out->set_dtype(meta.type_label());
			auto shape = node->shape();
			for (auto it = shape.begin(), et = shape.end(); it != et; ++it)
			{
				out->add_shape(*it);
			}
			if (auto ref = dynamic_cast<iDistRef*>(node.get()))
			{
				out->set_instance(ref->cluster_id());
			}
			else
			{
				out->set_instance(mgr->get_id());
			}
		}
		return grpc::Status::OK;
	}

	grpc::Status find_reachable (
		iDistManager* mgr,
		const FindReachableRequest& req,
		FindReachableResponse& res)
	{
		error::ErrptrT err = nullptr;
		auto& dests = req.dests();
		teq::TensSetT srcs;
		for (auto src : req.srcs())
		{
			auto local = mgr->lookup_node(err, src, false);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
			srcs.emplace(local.get());
		}
		auto reached = mgr->find_reachable(err, srcs,
			estd::StrSetT(dests.begin(), dests.end()));
		ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
		for (auto r : reached)
		{
			res.add_srcs(*mgr->lookup_id(r));
		}
		return grpc::Status::OK;
	}

	grpc::Status derive (
		iDistManager* mgr,
		const DeriveRequest& req,
		DeriveResponse& res)
	{
		auto& rgrads = req.root_grads();
		auto& targids = req.targets();

		error::ErrptrT err = nullptr;
		teq::GradMapT grads;
		teq::TensptrSetT parents;
		teq::TensptrSetT targets;
		teq::TensMapT<std::string> targmap;
		// populate grads and parents from request
		for (auto& reqpairs : rgrads)
		{
			auto local_id = reqpairs.first;
			auto grad_id = reqpairs.second;
			auto local = mgr->lookup_node(err, local_id, false);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
			auto grad = mgr->lookup_node(err, grad_id);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
			grads[local.get()].push_back(grad);
			parents.emplace(local);
		}
		for (auto targid : targids)
		{
			auto target = mgr->lookup_node(err, targid);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
			targets.emplace(target);
			targmap.emplace(target.get(), targid);
		}
		// run local derivation
		auto tgrads = mgr->derive(grads, parents, targets);
		// populate response grads
		teq::TensptrSetT to_track;
		for (auto targ : tgrads)
		{
			mgr->expose_node(targ.second);
		}
		auto resgrads = res.mutable_grads();
		for (auto targ : tgrads)
		{
			auto targid = targmap[targ.first];
			auto grad = targ.second;
			std::string gid = *mgr->lookup_id(grad.get());
			resgrads->insert(google::protobuf::MapPair<
				std::string,std::string>{targid, gid});
		}
		return grpc::Status::OK;
	}

	std::string alias_;

	AsyncService service_;

	std::unique_ptr<grpc::Server> server_;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::mutex rpc_exit_mutex_;

	std::condition_variable rpc_exit_;

	std::atomic<bool> rpc_done_ = false;
};

}

#endif // DISTRIB_SERVER_HPP
