
#include <grpcpp/grpcpp.h>

#ifndef DISTRIB_SERVER_HPP
#define DISTRIB_SERVER_HPP

#include "experimental/distrib/distr.grpc.pb.h"
#include "experimental/distrib/isession.hpp"

namespace distrib
{

struct DistrService final : public distr::DistrManager::Service
{
	DistrService (iDistribSess* sess) : sess_(sess) {}

	grpc::Status HealthCheck (grpc::ServerContext* context,
		const distr::Empty*, distr::Empty*) override
	{
		return grpc::Status::OK;
	}

	grpc::Status FindNodes (grpc::ServerContext* context,
		const distr::FindNodesRequest* req, distr::FindNodesResponse* res) override
	{
		auto& uuids = req->uuids();
		for (const std::string& uuid : uuids)
		{
			if (auto node = sess_->lookup_node(uuid, false))
			{
				distr::NodeMeta out;
				out.set_uuid(uuid);
				auto& meta = node->get_meta();
				out.set_dtype(meta.type_label());
				auto shape = node->shape();
				for (auto it = shape.begin(), et = shape.end(); it != et; ++it)
				{
					out.add_shape(*it);
				}
				if (auto ref = dynamic_cast<iDistRef*>(node.get()))
				{
					out.set_instance(ref->cluster_id());
				}
				else
				{
					out.set_instance(sess_->get_address());
				}
			}
		}
		return grpc::Status::OK;
	}

	grpc::Status GetData (grpc::ServerContext* context,
		const distr::GetDataRequest* req,
		grpc::ServerWriter<distr::NodeData>* writer) override
	{
		auto uuids = req->uuids();
		teq::TensSetT targets;
		std::unordered_map<std::string,teq::iTensor*> idtargets;
		for (const std::string& uuid : uuids)
		{
			auto tens = sess_->lookup_node(uuid).get();
			targets.emplace(tens);
			idtargets.emplace(uuid, tens);
		}
		eigen::Device device(std::numeric_limits<size_t>::max());
		sess_->update_target(device, targets);
		for (auto target : idtargets)
		{
			distr::NodeData data;
			data.set_uuid(target.first);
			auto tens = target.second;
			auto& meta = tens->get_meta();
			data.set_version(meta.state_version());

			void* raw = tens->device().data();
			auto dtype = (egen::_GENERATED_DTYPE) meta.type_code();

			size_t nelems = tens->shape().n_elems();
			google::protobuf::RepeatedField<double> field;
			field.Resize(nelems, 0);
			egen::type_convert(field.mutable_data(), raw, dtype, nelems);
			data.mutable_data()->Swap(&field);

			writer->Write(data);
		}
		return grpc::Status::OK;
	}

	grpc::Status ListPeers (grpc::ServerContext* context,
		const distr::Empty*, distr::ListPeersResponse* res) override
	{
		return grpc::Status::OK;
	}

	grpc::Status AddPeer (grpc::ServerContext* context,
		const distr::AddPeerRequest* req, distr::Empty*) override
	{
		return grpc::Status::OK;
	}

private:
	iDistribSess* sess_;
};

}

#endif // DISTRIB_SERVER_HPP
