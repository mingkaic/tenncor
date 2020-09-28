
#ifndef DISTR_HO_SERVICE_HPP
#define DISTR_HO_SERVICE_HPP

#include "tenncor/distr/iosvc/service.hpp"

#include "tenncor/hone/hone.hpp"
#include "tenncor/hone/hosvc/client.hpp"

namespace distr
{

namespace ho
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

using HoServiceT = DistrOptimization::AsyncService;

const std::string hosvc_key = "distr_hosvc";

struct DistrHoService final : public PeerService<DistrHoCli>
{
	DistrHoService (const PeerServiceConfig& cfg, io::DistrIOService* iosvc) :
		PeerService<DistrHoCli>(cfg), iosvc_(iosvc) {}

	teq::TensptrsT optimize (const teq::TensptrsT& roots,
		const opt::Optimization& optimize)
	{
		auto refs = distr::reachable_refs(roots);
		types::StrUMapT<types::StrUSetT> remotes;
		distr::separate_by_server(remotes, refs);

		types::StrUMapT<distr::iDistrRef*> refmap;
		for (auto ref : refs)
		{
			refmap.emplace(ref->node_id(), ref);
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
			// sort to make more predictable
			std::sort(node_ids.begin(), node_ids.end());

			PutOptimizeRequest req;
			req.mutable_uuids()->Swap(&node_ids);
			req.mutable_opts()->MergeFrom(optimize);
			completions.push_back(client->put_optimize(cq_, req,
			[&](PutOptimizeResponse& res)
			{
				auto opts = res.root_opts();
				for (auto& refpair : opts)
				{
					auto remote_ref = distr::io::node_meta_to_ref(refpair.second);
					static_cast<distr::DistrRef&>(*refmap.at(refpair.first)) =
						static_cast<distr::DistrRef&>(*remote_ref);
				}
			}));
		}

		while (false == completions.empty())
		{
			auto done = completions.front()->get_future();
			wait_on_future(done);
			if (done.valid())
			{
				if (auto err = done.get())
				{
					global::fatal(err->to_string());
				}
			}
			completions.pop_front();
		}
		return hone::optimize(roots, optimize);
	}

	void register_service (grpc::ServerBuilder& builder) override
	{
		builder.RegisterService(&service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		// PutOptimize
		auto popt_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:PutOptimize] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerCall<PutOptimizeRequest,
			PutOptimizeResponse>(popt_logger,
			[this](grpc::ServerContext* ctx, PutOptimizeRequest* req,
				grpc::ServerAsyncResponseWriter<PutOptimizeResponse>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestPutOptimize(ctx, req, writer, cq, ccq, tag);
			},
			[this](const PutOptimizeRequest& req, PutOptimizeResponse& res)
			{
				auto& uuids = req.uuids();
				auto& opts = req.opts();
				teq::TensptrsT roots;
				roots.reserve(uuids.size());
				for (const std::string& uuid : uuids)
				{
					error::ErrptrT err = nullptr;
					auto tens = this->iosvc_->lookup_node(err, uuid, false);
					_ERR_CHECK(err, grpc::NOT_FOUND, get_peer_id().c_str());
					roots.push_back(tens);
				}
				auto refresh = this->optimize(roots, opts);
				if (refresh.size() != roots.size())
				{
					std::string msg = fmts::sprintf(
						"[server %s] optimization lost nodes "
						"(input %d nodes, output %d nodes)",
						get_peer_id().c_str(), roots.size(), refresh.size());
					global::error(msg);
					return grpc::Status(grpc::INTERNAL, msg);
				}
				auto res_roots = res.mutable_root_opts();
				for (size_t i = 0, n = uuids.size(); i < n; ++i)
				{
					auto id = iosvc_->expose_node(refresh[i]);
					distr::io::NodeMeta meta;
					distr::io::tens_to_node_meta(meta, get_peer_id(), id, refresh[i]);
					res_roots->insert({uuids[i], meta});
				}
				return grpc::Status::OK;
			}, &cq);
	}

private:
	io::DistrIOService* iosvc_;

	DistrOptimization::AsyncService service_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_hosvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

ho::DistrHoService& get_hosvc (iDistrManager& manager);

}

#endif // DISTR_HO_SERVICE_HPP
