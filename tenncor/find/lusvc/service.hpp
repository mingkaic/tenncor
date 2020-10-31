
#ifndef DISTR_LU_SERVICE_HPP
#define DISTR_LU_SERVICE_HPP

#include "internal/query/query.hpp"

#include "tenncor/distr/iosvc/service.hpp"

#include "tenncor/find/lusvc/client.hpp"

namespace distr
{

namespace lu
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

const std::string lusvc_key = "distr_lusvc";

struct iLuService : public iService
{
	virtual ~iLuService (void) = default;

	virtual egrpc::RespondptrT<ListNodesResponse>
	make_list_nodes_responder (grpc::ServerContext& ctx) const = 0;

	SVC_RES_DECL(RequestListNodes, ListNodesRequest, ListNodesResponse)
};

struct LuService final : public iLuService
{
	grpc::Service* get_service (void) override
	{
		return &svc_;
	}

	egrpc::RespondptrT<ListNodesResponse>
	make_list_nodes_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<egrpc::GrpcResponder<ListNodesResponse>>(ctx);
	}

	SVC_RES_DEFN(RequestListNodes, ListNodesRequest, ListNodesResponse)

	DistrLookup::AsyncService svc_;
};

using OwnedSymbMapT = types::StrUMapT<teq::TensptrT>;

struct OwnedQueryResult
{
	operator teq::TensptrT() const
	{
		return root_;
	}

	friend bool operator == (const OwnedQueryResult& a, const OwnedQueryResult& b)
	{
		return a.root_ == b.root_ && a.symbs_ == b.symbs_;
	}

	teq::TensptrT root_;

	OwnedSymbMapT symbs_;
};

using OQResultsT = std::vector<OwnedQueryResult>;

struct DistrLuService final : public PeerService<DistrLuCli>
{
	DistrLuService (const PeerServiceConfig& cfg, io::DistrIOService* iosvc,
		CliBuildptrT builder =
			std::make_shared<ClientBuilder<DistrLuCli>>(),
		std::shared_ptr<iLuService> svc =
			std::make_shared<LuService>()) :
		PeerService<DistrLuCli>(cfg, builder),
		iosvc_(iosvc), service_(svc)
	{
		assert(nullptr != service_);
	}

	OQResultsT query (const teq::TensptrsT& roots,
		const query::Node& condition)
	{
		auto refs = distr::reachable_refs(roots);
		types::StrUMapT<types::StrUSetT> remotes;
		distr::separate_by_server(remotes, refs);

		types::StrUMapT<distr::iDistrRef*> refmap;
		for (auto ref : refs)
		{
			refmap.emplace(ref->node_id(), ref);
		}

		query::Query q;
		teq::multi_visit(q, roots);
		query::QResultsT local_matches = q.match(condition);
		auto owners = teq::track_ownptrs(roots);

		OQResultsT out;
		out.reserve(local_matches.size());
		for (auto& local : local_matches)
		{
			OwnedSymbMapT symbs;
			for (auto& lsymbs : local.symbs_)
			{
				symbs.emplace(lsymbs.first,
					owners.at(lsymbs.second));
			}
			out.push_back(OwnedQueryResult{
				owners.at(local.root_),
				symbs
			});
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

			ListNodesRequest req;
			req.mutable_uuids()->Swap(&node_ids);
			req.mutable_pattern()->MergeFrom(condition);
			completions.push_back(client->list_nodes(*cq_, req,
			[&](ListNodesResponse& res)
			{
				auto matches = res.matches();
				for (auto& match : matches)
				{
					auto& pb_symbs = match.symbs();
					auto root_ref = distr::io::node_meta_to_ref(match.root());
					OwnedSymbMapT symbs;
					for (auto& symbpair : pb_symbs)
					{
						symbs.insert({symbpair.first,
							distr::io::node_meta_to_ref(symbpair.second)});
					}
					out.push_back(OwnedQueryResult{
						root_ref,
						symbs
					});
				}
			}));
		}

		egrpc::wait_for(completions,
		[](error::ErrptrT err)
		{
			global::fatal(err->to_string());
		});
		return out;
	}

	void register_service (iServerBuilder& builder) override
	{
		builder.register_service(*service_);
	}

	void initialize_server_call (egrpc::iCQueue& cq) override
	{
		// ListNodes
		using ListNodesCallT = egrpc::AsyncServerCall<
			ListNodesRequest,ListNodesResponse>;
		auto popt_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:ListNodes] ",
				get_peer_id().c_str()));
		new ListNodesCallT(popt_logger,
		[this](grpc::ServerContext* ctx,
			ListNodesRequest* req,
			egrpc::iResponder<ListNodesResponse>& writer,
			egrpc::iCQueue& cq, void* tag)
		{
			this->service_->RequestListNodes(
				ctx, req, writer, cq, tag);
		},
		[this](const ListNodesRequest& req, ListNodesResponse& res)
		{
			auto& uuids = req.uuids();
			auto& pattern = req.pattern();
			teq::TensptrsT roots;
			roots.reserve(uuids.size());
			for (const std::string& uuid : uuids)
			{
				error::ErrptrT err = nullptr;
				auto tens = this->iosvc_->lookup_node(err, uuid, false);
				_ERR_CHECK(err, grpc::NOT_FOUND, get_peer_id().c_str());
				roots.push_back(tens);
			}
			auto results = this->query(roots, pattern);
			for (auto& result : results)
			{
				auto match = res.add_matches();
				auto id = iosvc_->expose_node(result.root_);
				auto pb_root = match->mutable_root();
				distr::io::tens_to_node_meta(*pb_root,
					get_peer_id(), id, result.root_);
				auto pb_symbs = match->mutable_symbs();
				for (auto& symbpair : result.symbs_)
				{
					auto symb_id = iosvc_->expose_node(symbpair.second);
					distr::io::NodeMeta pb_meta;
					distr::io::tens_to_node_meta(pb_meta,
						get_peer_id(), symb_id, symbpair.second);
					pb_symbs->insert({symbpair.first, pb_meta});
				}
			}
			return grpc::Status::OK;
		}, cq,
		[this](grpc::ServerContext& ctx)
		{
			return this->service_->make_list_nodes_responder(ctx);
		});
	}

private:
	io::DistrIOService* iosvc_;

	std::shared_ptr<iLuService> service_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_lusvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

lu::DistrLuService& get_lusvc (iDistrManager& manager);

}

#endif // DISTR_LU_SERVICE_HPP
