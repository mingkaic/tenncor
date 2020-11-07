
#ifndef DISTR_OX_SERVICE_HPP
#define DISTR_OX_SERVICE_HPP

#include "tenncor/serial/serial.hpp"

#include "tenncor/distr/iosvc/service.hpp"

#include "tenncor/serial/serial.hpp"
#include "tenncor/serial/oxsvc/client.hpp"
#include "tenncor/serial/oxsvc/topography.hpp"

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

const std::string oxsvc_key = "distr_serializesvc";

struct iSerializeService : public iService
{
	virtual ~iSerializeService (void) = default;

	virtual egrpc::RespondptrT<GetSaveGraphResponse>
	make_get_save_graph_responder (grpc::ServerContext& ctx) const = 0;

	virtual egrpc::RespondptrT<PostLoadGraphResponse>
	make_post_load_graph_responder (grpc::ServerContext& ctx) const = 0;

	SVC_RES_DECL(RequestGetSaveGraph, GetSaveGraphRequest, GetSaveGraphResponse)

	SVC_RES_DECL(RequestPostLoadGraph, PostLoadGraphRequest, PostLoadGraphResponse)
};

struct SerializeService final : public iSerializeService
{
	grpc::Service* get_service (void) override
	{
		return &svc_;
	}

	egrpc::RespondptrT<GetSaveGraphResponse>
	make_get_save_graph_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<egrpc::GrpcResponder<GetSaveGraphResponse>>(ctx);
	}

	egrpc::RespondptrT<PostLoadGraphResponse>
	make_post_load_graph_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<egrpc::GrpcResponder<PostLoadGraphResponse>>(ctx);
	}

	SVC_RES_DEFN(RequestGetSaveGraph, GetSaveGraphRequest, GetSaveGraphResponse)

	SVC_RES_DEFN(RequestPostLoadGraph, PostLoadGraphRequest, PostLoadGraphResponse)

	DistrSerialization::AsyncService svc_;
};

struct DistrSerializeService final : public PeerService<DistrSerializeCli>
{
	DistrSerializeService (const PeerServiceConfig& cfg, io::DistrIOService* iosvc,
		CliBuildptrT builder =
			std::make_shared<ClientBuilder<DistrSerializeCli>>(),
		std::shared_ptr<iSerializeService> svc =
			std::make_shared<SerializeService>()) :
		PeerService<DistrSerializeCli>(cfg, builder),
		iosvc_(iosvc), service_(svc)
	{
		assert(nullptr != service_);
	}

	TopographyT save_graph (onnx::GraphProto& pb_graph,
		const teq::TensptrsT& roots,
		onnx::TensptrIdT identified = {},
		teq::TensSetT stops = {})
	{
		TopographyT topo;
		auto refs = distr::reachable_refs(roots);
		types::StrUMapT<types::StrUSetT> remotes;
		distr::separate_by_server(remotes, refs);
		onnx::TensIdT local_identified;
		teq::TensSetT local_stops = stops;
		for (auto& id : identified)
		{
			local_identified.insert({id.left.get(), id.right});
		}
		for (auto& ref : refs)
		{
			local_identified.insert({ref, ref->node_id()});
			local_stops.emplace(ref);
		}

		types::StringsT idkeys;
		idkeys.reserve(identified.size());
		for (auto& id : identified)
		{
			idkeys.push_back(id.right);
		}
#ifdef ORDERED_SAVE
		std::sort(idkeys.begin(), idkeys.end());
#endif
		google::protobuf::Map<std::string,std::string> pb_identified;
		for (auto& id : idkeys)
		{
			auto node = identified.right.at(id);
			std::string ref_id = iosvc_->expose_node(node);
			pb_identified.insert({ref_id, id});
		}

		google::protobuf::RepeatedPtrField<std::string> pb_stop_ids;
		for (auto stop : stops)
		{
			auto stop_id = iosvc_->lookup_id(stop);
			if (false == bool(stop_id))
			{
				global::fatalf("can't find uuid for stop tensor %s",
					stop->to_string().c_str());
			}
			*pb_stop_ids.Add() = *stop_id;
		}

		std::list<egrpc::ErrPromiseptrT> completions;
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
#ifdef ORDERED_SAVE
			// sort to make more predictable
			std::sort(node_ids.begin(), node_ids.end());
#endif

			GetSaveGraphRequest req;
			req.mutable_uuids()->Swap(&node_ids);
			req.mutable_stop_uuids()->MergeFrom(pb_stop_ids);
			*req.mutable_identified() = pb_identified;

			completions.push_back(client->get_save_graph(*cq_, req,
			[&pb_graph, &topo](GetSaveGraphResponse& res)
			{
				auto& subgraph = res.graph();
				auto& subtopo = res.topography();
				merge_graph_proto(pb_graph, subgraph);
				merge_topograph(topo, subtopo);
			}));
			for (const auto& node : nodes)
			{
				if (estd::has(pb_identified, node))
				{
					topo.emplace(pb_identified.at(node), peer_id);
				}
				else
				{
					topo.emplace(node, peer_id);
				}
			}
		}

		egrpc::wait_for(completions,
		[](error::ErrptrT err)
		{
			global::fatal(err->to_string());
		});

		// add references to tens to avoid serialization of references
		serial::save_graph(pb_graph, roots, local_identified, local_stops);

		const auto& outputs = pb_graph.output();
		for (const auto& output : outputs)
		{
			topo.emplace(output.name(), get_peer_id()); // map each subgraph root set to peer id
		}
		return topo;
	}

	teq::TensptrsT load_graph (
		onnx::TensptrIdT& identified_tens,
		const onnx::GraphProto& pb_graph,
		TopographyT topography = TopographyT{})
	{
		std::string local_id = get_peer_id();

		// map unidentified roots to local peer id
		const auto& outputs = pb_graph.output();
		for (const auto& output : outputs)
		{
			topography.emplace(output.name(), local_id);
		}

		SegmentsT segments = split_topograph(pb_graph, topography);

		std::unordered_set<TopographicSeg*> rootsegs;
		std::transform(segments.begin(), segments.end(),
			std::inserter(rootsegs, rootsegs.end()),
			[](SegmentT seg){ return seg.get(); });

		SegmentsT segs;
		while (false == segments.empty())
		{
			auto seg = segments.front();
			segments.pop_front();
			segs.push_front(seg);
			for (auto& subgraph : seg->subgraphs_)
			{
				segments.push_back(subgraph.second);
			}
		}

		for (auto identified : identified_tens)
		{
			iosvc_->expose_node(identified.left);
		}

		teq::TensptrsT roots;
		types::StrUMapT<std::string> idrefs;
		for (auto& seg : segs)
		{
			error::ErrptrT err = nullptr;
			auto peer_id = seg->color_;
			auto& subgraph = seg->graph_;
			const auto& subs = seg->subgraphs_;
			types::StrUSetT refs;
			for (const auto& sub : subs)
			{
				const auto& outputs = sub.second->graph_.output();
				for (const auto& output : outputs)
				{
					auto outname = output.name();
					refs.emplace(estd::must_getf(idrefs, outname,
						"cannot find reference for %s", outname));
				}
			}
			types::StrUMapT<std::string> local_refs;
			if (local_id == peer_id)
			{
				local_refs = local_load_graph(
					err, identified_tens, subgraph, refs);
				if (nullptr != err)
				{
					global::fatal(err->to_string());
				}
			}
			else
			{
				auto client = get_client(err, peer_id);
				if (nullptr != err)
				{
					global::fatal(err->to_string());
				}
				google::protobuf::RepeatedPtrField<std::string> pb_refs(
					refs.begin(), refs.end());
				PostLoadGraphRequest req;
				req.mutable_graph()->MergeFrom(subgraph);
				req.mutable_refs()->Swap(&pb_refs);
				auto done = client->post_load_graph(*cq_, req,
				[&](PostLoadGraphResponse& res)
				{
					auto& pb_roots = res.roots();
					for (auto& rootpair : pb_roots)
					{
						local_refs.emplace(rootpair);
					}
				});
				egrpc::wait_for(*done,
				[](error::ErrptrT err)
				{
					global::fatal(err->to_string());
				});
			}
			idrefs.insert(local_refs.begin(), local_refs.end());
			if (estd::has(rootsegs, seg.get()))
			{
				for (auto local_ref : local_refs)
				{
					auto root = iosvc_->lookup_node(err, local_ref.second);
					if (err != nullptr)
					{
						global::fatal(err->to_string());
					}
					roots.push_back(root);
					identified_tens.insert({root, local_ref.first});
				}
			}
		}
		return roots;
	}

	void register_service (iServerBuilder& builder) override
	{
		builder.register_service(*service_);
	}

	void initialize_server_call (egrpc::iCQueue& cq) override
	{
		// GetSaveGraph
		using GetSaveGraphCallT = egrpc::AsyncServerCall<
			GetSaveGraphRequest,GetSaveGraphResponse>;
		auto gsave_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:GetSaveGraph] ",
				get_peer_id().c_str()));
		new GetSaveGraphCallT(gsave_logger,
		[this](grpc::ServerContext* ctx,
			GetSaveGraphRequest* req,
			egrpc::iResponder<GetSaveGraphResponse>& writer,
			egrpc::iCQueue& cq, void* tag)
		{
			this->service_->RequestGetSaveGraph(
				ctx, req, writer, cq, tag);
		},
		[this](const GetSaveGraphRequest& req, GetSaveGraphResponse& res)
		{
			error::ErrptrT err = nullptr;

			auto& uuids = req.uuids();
			auto& ids = req.identified();
			auto& stop_ids = req.stop_uuids();

			teq::TensptrsT roots;
			roots.reserve(uuids.size());

			teq::TensSetT stops;
			stops.reserve(stop_ids.size());

			onnx::TensptrIdT identified;
			for (auto& id : ids)
			{
				auto tens = this->iosvc_->lookup_node(err, id.first);
				identified.insert({tens, id.second});
			}

			for (const std::string& uuid : uuids)
			{
				auto tens = this->iosvc_->lookup_node(err, uuid, false);
				_ERR_CHECK(err, grpc::NOT_FOUND, get_peer_id().c_str());
				roots.push_back(tens);
				identified.insert({tens, uuid});
			}

			for (const std::string& suuid : stop_ids)
			{
				auto tens = this->iosvc_->lookup_node(err, suuid);
				_ERR_CHECK(err, grpc::NOT_FOUND, get_peer_id().c_str());
				stops.emplace(tens.get());
			}

			auto topo = this->save_graph(
				*res.mutable_graph(), roots, identified, stops);
			auto outopo = res.mutable_topography();
			for (auto& entry : topo)
			{
				outopo->insert({entry.first, entry.second});
			}
			return grpc::Status::OK;
		}, cq,
		[this](grpc::ServerContext& ctx)
		{
			return this->service_->make_get_save_graph_responder(ctx);
		});

		// PostLoadGraph
		using PostLoadGraphCallT = egrpc::AsyncServerCall<
			PostLoadGraphRequest,PostLoadGraphResponse>;
		auto pload_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:PostLoadGraph] ",
				get_peer_id().c_str()));
		new PostLoadGraphCallT(pload_logger,
		[this](grpc::ServerContext* ctx,
			PostLoadGraphRequest* req,
			egrpc::iResponder<PostLoadGraphResponse>& writer,
			egrpc::iCQueue& cq, void* tag)
		{
			this->service_->RequestPostLoadGraph(
				ctx, req, writer, cq, tag);
		},
		[this](const PostLoadGraphRequest& req, PostLoadGraphResponse& res)
		{
			onnx::TensptrIdT identified;
			auto& reqgraph = req.graph();
			auto& refs = req.refs();
			error::ErrptrT err = nullptr;
			auto local_refs = this->local_load_graph(err, identified, reqgraph,
				types::StrUSetT(refs.begin(), refs.end()));
			_ERR_CHECK(err, grpc::NOT_FOUND, get_peer_id().c_str());
			auto res_roots = res.mutable_roots();
			for (auto local_ref : local_refs)
			{
				res_roots->insert({local_ref.first, local_ref.second});
			}
			return grpc::Status::OK;
		}, cq,
		[this](grpc::ServerContext& ctx)
		{
			return this->service_->make_post_load_graph_responder(ctx);
		});
	}

private:
	types::StrUMapT<std::string> local_load_graph (error::ErrptrT& err,
		onnx::TensptrIdT& identified_tens, const onnx::GraphProto& subgraph,
		const types::StrUSetT& refs)
	{
		err = nullptr;
		std::string local_id = get_peer_id();
		onnx::TensptrIdT identified = identified_tens;
		for (auto ref : refs)
		{
			auto node = iosvc_->lookup_node(err, ref);
			if (nullptr != err)
			{
				return {};
			}
			identified.insert({node, ref});
		}
		auto roots = serial::load_graph(identified, subgraph);
		types::StrUMapT<std::string> idrefs;
		for (auto root : roots)
		{
			auto root_id = estd::must_getf(identified.left, root,
				"couldn't find id for root %p", root.get());
			auto root_refid = iosvc_->expose_node(root, root_id);
			idrefs.emplace(root_id, root_refid);
		}
		return idrefs;
	}

	io::DistrIOService* iosvc_;

	std::shared_ptr<iSerializeService> service_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_oxsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

ox::DistrSerializeService& get_oxsvc (iDistrManager& manager);

}

#endif // DISTR_OX_SERVICE_HPP
