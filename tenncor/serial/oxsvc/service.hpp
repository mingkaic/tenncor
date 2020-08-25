
#include "egrpc/egrpc.hpp"

#include "tenncor/serial/serial.hpp"

#include "tenncor/distrib/iosvc/service.hpp"

#include "tenncor/serial/serial.hpp"
#include "tenncor/serial/oxsvc/client.hpp"

#ifndef DISTRIB_OX_SERVICE_HPP
#define DISTRIB_OX_SERVICE_HPP

namespace distr
{

namespace ox
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

// copy everything from ingraph inserting into outgraph
// except for output
void merge_graph_proto (onnx::GraphProto& outgraph,
	const onnx::GraphProto& ingraph);

const std::string oxsvc_key = "distr_serializesvc";

// Topographic map of node id to peer id where node should be deploy to
using TopographyT = types::StrUMapT<std::string>;

struct DistrSerializeService final : public PeerService<DistrSerializeCli>
{
	DistrSerializeService (const PeerServiceConfig& cfg, io::DistrIOService* iosvc) :
		PeerService<DistrSerializeCli>(cfg), iosvc_(iosvc) {}

	template <typename TS> // todo: use concept tensptr_range
	void save_graph (onnx::GraphProto& pb_graph,
		const TS& roots, onnx::TensIdT identified = {})
	{
		auto refs = distr::reachable_refs(roots);
		for (auto& ref : refs)
		{
			identified.insert({ref, ref->node_id()});
		}
		types::StrUMapT<types::StrUSetT> remotes;
		distr::separate_by_server(remotes, refs);

		for (auto& remote : remotes)
		{
			auto peer_id = remote.first;
			auto nodes = remote.second;
			error::ErrptrT err = nullptr;
			auto client = get_client(err, peer_id);
			if (nullptr != err)
			{
				global::fatal(err->to_string());
			}
			google::protobuf::RepeatedPtrField<std::string>
			node_ids(nodes.begin(), nodes.end());

			GetSaveGraphRequest req;
			req.mutable_uuids()->Swap(&node_ids);
			client->get_save_graph(cq_, req,
			[&pb_graph](GetSaveGraphResponse& res)
			{
				auto& subgraph = res.graph();
				merge_graph_proto(pb_graph, subgraph);
			});
		}

		// add references to tens to avoid serialization of references
		serial::save_graph(pb_graph, roots, identified);
	}

	teq::TensptrsT load_graph (onnx::TensptrIdT& identified_tens,
		const onnx::GraphProto& pb_graph,
		const TopographyT& topography = TopographyT{})
	{
		auto& pb_nodes = pb_graph.node();
		std::string peer_id;
		types::StrUMapT<types::StrUSetT> remotes;
		for (auto& pb_node : pb_nodes)
		{
			auto id = pb_node.name();
			// lookup to see if node needs to be deployed remotely
			if (estd::get(peer_id, topography, id))
			{
				remotes[peer_id].emplace(id);
			}
		}
		for (auto& remote : remotes)
		{
			auto peer_id = remote.first;
			auto nodes = remote.second;
			error::ErrptrT err = nullptr;
			auto client = get_client(err, peer_id);
			if (nullptr != err)
			{
				global::fatal(err->to_string());
			}

			google::protobuf::Map<std::string,std::string> pb_topo(
				topography.begin(), topography.end());

			PostLoadGraphRequest req;
			auto reqgraph = req.mutable_graph();
			req.mutable_topography()->swap(pb_topo);
			merge_graph_proto(*reqgraph, pb_graph);
			for (auto node : nodes)
			{
				onnx::ValueInfoProto* pb_output = reqgraph->add_output();
				pb_output->set_name(node);
			}
			client->post_load_graph(cq_, req,
			[&identified_tens](PostLoadGraphResponse& res)
			{
				auto& refmetas = res.values();
				for (auto& refmeta : refmetas)
				{
					auto ref = ::distr::io::node_meta_to_ref(refmeta);
					identified_tens.insert({ref, refmeta.uuid()});
				}
			});
		}

		return serial::load_graph(identified_tens, pb_graph);
	}

	void register_service (grpc::ServerBuilder& builder) override
	{
		builder.RegisterService(&service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		// GetSaveGraph
		auto gsave_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:GetSaveGraph] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerCall<GetSaveGraphRequest,
			GetSaveGraphResponse>(gsave_logger,
			[this](grpc::ServerContext* ctx, GetSaveGraphRequest* req,
				grpc::ServerAsyncResponseWriter<GetSaveGraphResponse>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestGetSaveGraph(ctx, req, writer, cq, ccq, tag);
			},
			[this](
				const GetSaveGraphRequest& req,
				GetSaveGraphResponse& res)
			{
				auto& uuids = req.uuids();
				teq::TensptrsT roots;
				roots.reserve(uuids.size());
				onnx::TensIdT identified;
				for (const std::string& uuid : uuids)
				{
					error::ErrptrT err = nullptr;
					auto tens = this->iosvc_->lookup_node(err, uuid, false);
					_ERR_CHECK(err, grpc::NOT_FOUND, get_peer_id().c_str());
					roots.push_back(tens);
					identified.insert({tens.get(), uuid});
				}
				this->save_graph(*res.mutable_graph(), roots, identified);
				return grpc::Status::OK;
			}, &cq);

		// PostLoadGraph
		auto pload_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:PostLoadGraph] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerCall<PostLoadGraphRequest,
			PostLoadGraphResponse>(pload_logger,
			[this](grpc::ServerContext* ctx, PostLoadGraphRequest* req,
				grpc::ServerAsyncResponseWriter<PostLoadGraphResponse>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestPostLoadGraph(
					ctx, req, writer, cq, ccq, tag);
			},
			[this](
				const PostLoadGraphRequest& req,
				PostLoadGraphResponse& res)
			{
				onnx::TensptrIdT identified;
				auto& reqgraph = req.graph();
				auto& topo = req.topography();
				auto roots = this->load_graph(identified, reqgraph,
					TopographyT(topo.begin(), topo.end()));
				// expose roots
				for (auto root : roots)
				{
					auto id = iosvc_->expose_node(root);
					auto alias = identified.left.at(root);
					iosvc_->set_alias(alias, id);
					auto meta = res.add_values();
					::distr::io::tens_to_node_meta(
						*meta, get_peer_id(), alias, root);
				}
				return grpc::Status::OK;
			}, &cq);
	}

private:
	io::DistrIOService* iosvc_;

	DistrSerialization::AsyncService service_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_oxsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

ox::DistrSerializeService& get_oxsvc (iDistrManager& manager);

}

#endif // DISTRIB_OX_SERVICE_HPP
