
#include <condition_variable>
#include <mutex>
#include <thread>

#include "distrib/async.hpp"
#include "distrib/isession.hpp"

#ifndef DISTRIB_SERVER_HPP
#define DISTRIB_SERVER_HPP

namespace distrib
{

struct DistrService final : public distr::DistrManager::Service
{
	DistrService (iDistribSess* sess) : sess_(sess) {}

	grpc::Status FindNodes (grpc::ServerContext* context,
		const distr::FindNodesRequest* req, distr::FindNodesResponse* res) override
	{
		auto& uuids = req->uuids();
		for (const std::string& uuid : uuids)
		{
			err::ErrptrT err;
			if (auto node = sess_->lookup_node(err, uuid, false))
			{
				distr::NodeMeta* out = res->add_values();
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
					out->set_instance(sess_->get_id());
				}
			}
			else
			{
				teq::errorf("server node lookup failure: %s",
					err->to_string().c_str());
				return grpc::Status(grpc::NOT_FOUND, err->to_string());
			}
		}
		return grpc::Status::OK;
	}

	grpc::Status GetData (grpc::ServerContext* context,
		const distr::GetDataRequest* req,
		grpc::ServerWriter<distr::NodeData>* writer) override
	{
		auto& uuids = req->uuids();
		teq::TensSetT targets;
		std::unordered_map<std::string,
			std::pair<teq::iTensor*,size_t>> prev_states;
		for (const std::string& uuid : uuids)
		{
			err::ErrptrT err;
			auto tens = sess_->lookup_node(err, uuid, false).get();
			if (nullptr == tens)
			{
				teq::errorf("failed to find node %s: %s",
					uuid.c_str(), err->to_string().c_str());
				return grpc::Status(grpc::NOT_FOUND, err->to_string());
			}
			targets.emplace(tens);
			prev_states.emplace(uuid, std::pair<teq::iTensor*,size_t>{
				tens, tens->get_meta().state_version()});
		}
		eigen::Device device(std::numeric_limits<size_t>::max());
		sess_->update_target(device, targets);
		for (auto target : prev_states)
		{
			auto id = target.first;
			auto tens = target.second.first;
			auto prev_vers = target.second.second;

			auto& meta = tens->get_meta();
			size_t latest = meta.state_version();

			if (prev_vers != latest)
			{
				distr::NodeData data;
				data.set_uuid(id);
				data.set_version(latest);

				void* raw = tens->device().data();
				auto dtype = (egen::_GENERATED_DTYPE) meta.type_code();

				size_t nelems = tens->shape().n_elems();
				google::protobuf::RepeatedField<double> field;
				field.Resize(nelems, 0);
				egen::type_convert(field.mutable_data(), raw, dtype, nelems);
				data.mutable_data()->Swap(&field);

				writer->Write(data);
			}
		}
		return grpc::Status::OK;
	}

private:
	iDistribSess* sess_;
};

struct DistrServer final
{
	// non-blocking
	DistrServer (iDistribSess* sess, size_t port) :
		service_(sess)
	{
		std::string address = fmts::sprintf("0.0.0.0:%d", port);
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		builder.RegisterService(&service_);
		server_ = builder.BuildAndStart();

		teq::infof("server listening on %s", address.c_str());
	}

	~DistrServer (void)
	{
		server_->Shutdown();
	}

private:
	DistrService service_;

	std::unique_ptr<grpc::Server> server_;
};

struct AsyncDistrServer final
{
	using AsyncService = distr::DistrManager::AsyncService;

	// non-blocking
	AsyncDistrServer (iDistribSess* sess, size_t port)
	{
		std::string address = fmts::sprintf("0.0.0.0:%d", port);
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		builder.RegisterService(&service_);
		cq_ = builder.AddCompletionQueue();
		server_ = builder.BuildAndStart();

		teq::infof("server listening on %s", address.c_str());

		std::thread rpc_job(&AsyncDistrServer::handle_rpcs, this, sess);
		rpc_job.detach();
	}

	~AsyncDistrServer (void)
	{
		server_->Shutdown();
		cq_->Shutdown();
		std::unique_lock<std::mutex> lk(rpc_exit_mutex_);
		rpc_exit_.wait(lk, [this]{ return bool(this->rpc_done_); });
	}

private:
	using DataStatesT = std::unordered_map<std::string,
		std::pair<teq::iTensor*,size_t>>;

	// This can be run in multiple threads if needed.
	void handle_rpcs (iDistribSess* sess)
	{
		// FindNodes
		new AsyncServerCall<distr::FindNodesRequest,distr::FindNodesResponse>(
		[this](grpc::ServerContext* ctx, distr::FindNodesRequest* req,
			grpc::ServerAsyncResponseWriter<distr::FindNodesResponse>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_.RequestFindNodes(ctx, req, writer, cq, ccq, tag);
		},
		[sess](const distr::FindNodesRequest& req, distr::FindNodesResponse& res) -> grpc::Status
		{
			auto& uuids = req.uuids();
			for (const std::string& uuid : uuids)
			{
				err::ErrptrT err;
				if (auto node = sess->lookup_node(err, uuid, false))
				{
					distr::NodeMeta* out = res.add_values();
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
				else
				{
					teq::errorf("server node lookup failure: %s",
						err->to_string().c_str());
					return grpc::Status(grpc::NOT_FOUND, err->to_string());
				}
			}
			return grpc::Status::OK;
		}, cq_.get());

		// GetData
		new AsyncServerStreamCall<
			distr::GetDataRequest,distr::NodeData,DataStatesT>(
		[this](grpc::ServerContext* ctx, distr::GetDataRequest* req,
			grpc::ServerAsyncWriter<distr::NodeData>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_.RequestGetData(ctx, req, writer, cq, ccq, tag);
		},
		[sess](DataStatesT& prev_states, const distr::GetDataRequest& req) -> grpc::Status
		{
			auto& uuids = req.uuids();
			teq::TensSetT targets;
			for (const std::string& uuid : uuids)
			{
				err::ErrptrT err;
				auto tens = sess->lookup_node(err, uuid, false).get();
				if (nullptr == tens)
				{
					teq::errorf("failed to find node %s: %s",
						uuid.c_str(), err->to_string().c_str());
					return grpc::Status(grpc::NOT_FOUND, err->to_string());
				}
				targets.emplace(tens);
				prev_states.emplace(uuid, std::pair<teq::iTensor*,size_t>{
					tens, tens->get_meta().state_version()});
			}
			eigen::Device device(std::numeric_limits<size_t>::max());
			sess->update_target(device, targets);
			return grpc::Status::OK;
		},
		[](const distr::GetDataRequest& req, DataStatesT::iterator& it,
			distr::NodeData& reply)
		{
			auto id = it->first;
			auto tens = it->second.first;
			auto prev_vers = it->second.second;

			auto& meta = tens->get_meta();
			size_t latest = meta.state_version();

			if (prev_vers != latest)
			{
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
			}
			return false;
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

	AsyncService service_;

	std::unique_ptr<grpc::Server> server_;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::mutex rpc_exit_mutex_;

	std::condition_variable rpc_exit_;

	std::atomic<bool> rpc_done_ = false;
};

}

#endif // DISTRIB_SERVER_HPP
