
#ifndef DISTR_IO_SERVICE_HPP
#define DISTR_IO_SERVICE_HPP

#include "tenncor/distr/imanager.hpp"
#include "tenncor/distr/iosvc/client.hpp"
#include "tenncor/distr/iosvc/data.hpp"

namespace distr
{

namespace io
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

const std::string iosvc_key = "distr_iosvc";

struct iIOService : public iService
{
	virtual ~iIOService (void) = default;

	virtual egrpc::RespondptrT<ListNodesResponse>
	make_list_nodes_responder (grpc::ServerContext& ctx) const = 0;

	SVC_RES_DECL(RequestListNodes, ListNodesRequest, ListNodesResponse)
};

struct IOService final : public iIOService
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

	DistrInOut::AsyncService svc_;
};

struct DistrIOService final : public PeerService<DistrIOCli>
{
	DistrIOService (const PeerServiceConfig& cfg,
		CliBuildptrT builder =
			std::make_shared<ClientBuilder<DistrIOCli>>(),
		std::shared_ptr<iIOService> svc =
			std::make_shared<IOService>()) :
		PeerService<DistrIOCli>(cfg, builder),
		service_(svc), data_(cfg.p2p_)
	{
		assert(nullptr != service_);
	}

	std::string expose_node (teq::TensptrT tens,
		const OptIDT& suggest_id = OptIDT())
	{
		return data_.cache_tens(tens, suggest_id);
	}

	OptIDT lookup_id (teq::iTensor* tens) const
	{
		return data_.get_id(tens);
	}

	teq::TensptrT lookup_node (error::ErrptrT& err,
		const std::string& id, bool recursive = true)
	{
		err = nullptr;
		if (auto out = data_.get_tens(id))
		{
			return out;
		}
		if (false == recursive)
		{
			err = error::errorf(
				"no id %s found locally: will not recurse", id.c_str());
			return nullptr;
		}

		auto peer_id = data_.get_peer(id);
		if (false == bool(peer_id))
		{
			err = error::errorf("no peer found for node %s", id.c_str());
			return nullptr;
		}

		auto client = get_client(err, *peer_id);
		if (nullptr != err)
		{
			return nullptr;
		}

		ListNodesRequest req;
		teq::TensptrT ref = nullptr;
		req.add_uuids(id);
		auto done = client->list_nodes(*cq_, req,
			[&, this](ListNodesResponse& res)
			{
				if (res.values().empty())
				{
					err = error::errorf("no result found in received peer '%s'",
						peer_id->c_str());
					return;
				}
				auto node = res.values().at(0);
				ref = this->lookup_or_expose_ref(node);
			});
		egrpc::wait_for(*done,
		[&err](error::ErrptrT inerr)
		{
			err = inerr;
		});
		return ref;
	}

	teq::TensptrT must_lookup_node (
		const std::string& id,
		bool recursive = true)
	{
		error::ErrptrT err = nullptr;
		auto out = lookup_node(err, id, recursive);
		if (nullptr != err)
		{
			global::fatal(err->to_string());
		}
		return out;
	}

	std::string must_lookup_id (teq::iTensor* tens)
	{
		auto id = lookup_id(tens);
		if (false != bool(id))
		{
			global::fatalf("failed to find '%s'[%p]", tens->to_string().c_str(), tens);
		}
		return *id;
	}

	teq::TensptrT lookup_or_expose_ref (
		const NodeMeta& ref)
	{
		auto lookup_ref = ref.uuid();
		if (auto tens = data_.get_tens(lookup_ref))
		{
			return tens;
		}
		auto tens = node_meta_to_ref(ref);
		expose_node(tens);
		return tens;
	}

	DRefptrSetT get_remotes (void) const
	{
		return data_.get_remotes();
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
		auto lnodes_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:ListNodes] ",
			get_peer_id().c_str()));
		new ListNodesCallT(lnodes_logger,
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
			auto alias = fmts::sprintf("%s:ListNodes", get_peer_id().c_str());
			auto& uuids = req.uuids();
			for (const std::string& uuid : uuids)
			{
				error::ErrptrT err = nullptr;
				auto tens = lookup_node(err, uuid, false);
				_ERR_CHECK(err, grpc::NOT_FOUND, alias.c_str());

				NodeMeta* out = res.add_values();
				tens_to_node_meta(*out, get_peer_id(), uuid, tens);
			}
			return grpc::Status::OK;
		}, cq,
		[this](grpc::ServerContext& ctx)
		{
			return this->service_->make_list_nodes_responder(ctx);
		});
	}

private:
	std::shared_ptr<iIOService> service_;

	DistrIOData data_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_iosvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

io::DistrIOService& get_iosvc (iDistrManager& manager);

}

#endif // DISTR_IO_SERVICE_HPP
