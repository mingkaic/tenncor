
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

struct DistrSerializeService final : public PeerService<DistrSerializeCli>
{
	DistrSerializeService (const PeerServiceConfig& cfg, io::DistrIOService* iosvc) :
		PeerService<DistrSerializeCli>(cfg), iosvc_(iosvc) {}

	TopographyT save_graph (onnx::GraphProto& pb_graph,
		const teq::TensptrsT& roots, onnx::TensIdT identified = {})
	{
		TopographyT topo;
		auto refs = distr::reachable_refs(roots);
		types::StrUMapT<types::StrUSetT> remotes;
		distr::separate_by_server(remotes, refs);
		teq::TensSetT stops;
		for (auto& ref : refs)
		{
			identified.insert({ref, ref->node_id()});
			stops.emplace(ref);
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
			completions.push_back(client->get_save_graph(cq_, req,
			[&pb_graph, &topo](GetSaveGraphResponse& res)
			{
				auto& subgraph = res.graph();
				auto& subtopo = res.topography();
				merge_graph_proto(pb_graph, subgraph);
				merge_topograph(topo, subtopo);
			}));
			for (const auto& node : nodes)
			{
				topo.emplace(node, peer_id);
			}
		}

		egrpc::wait_for(completions,
		[](error::ErrptrT err)
		{
			global::fatal(err->to_string());
		});

		// add references to tens to avoid serialization of references
		serial::save_graph(pb_graph, roots, identified, stops);

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
					refs.emplace(output.name());
				}
			}
			teq::TensptrsT local_roots;
			if (local_id == peer_id)
			{
				local_roots = local_load_graph(
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
				auto done = client->post_load_graph(cq_, req);
				egrpc::wait_for(*done,
				[](error::ErrptrT err)
				{
					global::fatal(err->to_string());
				});

				const auto& suboutputs = subgraph.output();
				for (const auto& suboutput : suboutputs)
				{
					std::string id = suboutput.name();
					auto root = iosvc_->lookup_node(err, id);
					local_roots.push_back(root);
				}
			}
			if (estd::has(rootsegs, seg.get()))
			{
				roots.insert(roots.end(),
					local_roots.begin(), local_roots.end());
			}
		}
		return roots;
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
				auto topo = this->save_graph(*res.mutable_graph(), roots, identified);
				auto outopo = res.mutable_topography();
				for (auto& entry : topo)
				{
					outopo->insert({entry.first, entry.second});
				}
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
				auto& refs = req.refs();
				error::ErrptrT err = nullptr;
				this->local_load_graph(err, identified, reqgraph,
					types::StrUSetT(refs.begin(), refs.end()));
				_ERR_CHECK(err, grpc::NOT_FOUND, get_peer_id().c_str());
				return grpc::Status::OK;
			}, &cq);
	}

private:
	teq::TensptrsT local_load_graph (error::ErrptrT& err,
		onnx::TensptrIdT& identified_tens,
		const onnx::GraphProto& subgraph, const types::StrUSetT& refs)
	{
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
		const auto& suboutputs = subgraph.output();
		for (const auto& suboutput : suboutputs)
		{
			std::string id = suboutput.name();
			assert(id == iosvc_->expose_node(
				identified.right.at(id), id));
		}
		identified_tens.insert(identified.begin(), identified.end());
		return roots;
	}

	io::DistrIOService* iosvc_;

	DistrSerialization::AsyncService service_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_oxsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

ox::DistrSerializeService& get_oxsvc (iDistrManager& manager);

}

#endif // DISTR_OX_SERVICE_HPP
