
#include <condition_variable>
#include <mutex>
#include <thread>

#include "distrib/async.hpp"
#include "distrib/isession.hpp"

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

using PartDerF = std::function<teq::TensMapT<teq::TensptrT>(
	teq::GradMapT&, iDistribSess*,
	const teq::TensptrSetT&, const teq::TensptrSetT&,const DRefptrSetT&)>;

struct DistrServer final
{
	using AsyncService = DistrManager::AsyncService;

	// non-blocking
	DistrServer (iDistribSess* sess, PartDerF derive,
		size_t port, std::string alias) :
		derive_(derive), alias_(alias)
	{
		std::string address = fmts::sprintf("0.0.0.0:%d", port);
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		builder.RegisterService(&service_);
		cq_ = builder.AddCompletionQueue();
		server_ = builder.BuildAndStart();

		teq::infof("[server %s] listening on %s", alias_.c_str(), address.c_str());

		std::thread rpc_job(&DistrServer::handle_rpcs, this, sess);
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
	void handle_rpcs (iDistribSess* sess)
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
		[this, sess](
			const FindNodesRequest& req,
			FindNodesResponse& res)
		{
			return this->find_nodes(sess, req, res);
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
		[this, sess](
			const DeriveRequest& req,
			DeriveResponse& res)
		{
			return this->derive(sess, req, res);
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
		[this, sess](DataStatesT& states, const GetDataRequest& req) -> grpc::Status
		{
			auto& uuids = req.uuids();
			teq::TensSetT targets;
			for (const std::string& uuid : uuids)
			{
				error::ErrptrT err = nullptr;
				auto tens = sess->lookup_node(err, uuid, false).get();
				ERR_CHECK(err, grpc::NOT_FOUND, this->alias_.c_str());
				targets.emplace(tens);
				states.emplace(uuid, tens);
			}
			eigen::Device device(std::numeric_limits<size_t>::max());
			sess->update_target(device, targets);
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

	grpc::Status find_nodes (iDistribSess* sess,
		const FindNodesRequest& req,
		FindNodesResponse& res)
	{
		auto& uuids = req.uuids();
		for (const std::string& uuid : uuids)
		{
			error::ErrptrT err = nullptr;
			auto node = sess->lookup_node(err, uuid, false);
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
				out->set_instance(sess->get_id());
			}
		}
		return grpc::Status::OK;
	}

	grpc::Status derive (iDistribSess* sess,
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
		DRefptrSetT reachables;
		for (auto& reqpairs : rgrads)
		{
			auto local_id = reqpairs.first;
			auto grad_id = reqpairs.second;
			auto local = sess->lookup_node(err, local_id, false);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
			auto grad = sess->lookup_node(err, grad_id);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
			grads[local.get()].push_back(grad);
			parents.emplace(local);
			auto rec = reachable_refs(local);
			reachables.insert(rec.begin(), rec.end());
		}
		for (auto targid : targids)
		{
			auto target = sess->lookup_node(err, targid);
			ERR_CHECK(err, grpc::NOT_FOUND, alias_.c_str());
			targets.emplace(target);
			targmap.emplace(target.get(), targid);
		}
		// run local derivation
		auto tgrads = derive_(
			grads, sess, parents, targets, reachables);
		// populate response grads
		teq::TensptrSetT to_track;
		for (auto targ : tgrads)
		{
			to_track.emplace(targ.second);
		}
		sess->track(to_track); // track to expose the gradient tensors
		auto resgrads = res.mutable_grads();
		for (auto targ : tgrads)
		{
			auto targid = targmap[targ.first];
			auto grad = targ.second;
			std::string gid = *sess->lookup_id(grad);
			resgrads->insert(google::protobuf::MapPair<
				std::string,std::string>{targid, gid});
		}
		return grpc::Status::OK;
	}

	std::string alias_;

	AsyncService service_;

	std::unique_ptr<grpc::Server> server_;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	PartDerF derive_;

	std::mutex rpc_exit_mutex_;

	std::condition_variable rpc_exit_;

	std::atomic<bool> rpc_done_ = false;
};

}

#endif // DISTRIB_SERVER_HPP
