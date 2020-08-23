
#include "tenncor/eteq/eteq.hpp"

#include "tenncor/distrib/iosvc/service.hpp"
#include "tenncor/eteq/opsvc/client.hpp"

#ifndef DISTRIB_OP_SERVICE_HPP
#define DISTRIB_OP_SERVICE_HPP

namespace distr
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

#define _MAKE_DERFUNC(REAL_TYPE)\
builder = new eteq::DerivativeFuncs<REAL_TYPE>();

using OpServiceT = op::DistrOperation::AsyncService;

using DataStatesT = std::unordered_map<std::string,teq::iTensor*>;

const std::string opsvc_key = "distr_opsvc";

struct BackpropMeta
{
	// teq::TensptrT root_;

	teq::TensSetT targets_;
};

bool process_get_data (
	const op::GetDataRequest& req,
	DataStatesT::iterator& it,
	op::NodeData& reply);

struct DistrOpService final : public PeerService<DistrOpCli>
{
	DistrOpService (const PeerServiceConfig& cfg, DistrIOService* iosvc) :
		PeerService<DistrOpCli>(cfg), iosvc_(iosvc) {}

	/// Evalute target tensor set ignoring all tensors in ignored set
	void evaluate (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored)
	{
		// find all reachable refs and make remote call
		auto refs = reachable_refs(targets, ignored);
		types::StrUMapT<types::StrUSetT> servers;
		separate_by_server(servers, refs);
		std::list<egrpc::ErrPromiseptrT> completions;
		for (auto& spair : servers)
		{
			auto peer_id = spair.first;
			auto nodes = spair.second;
			error::ErrptrT err = nullptr;
			auto client = get_client(err, peer_id);
			if (nullptr != err)
			{
				global::fatal(err->to_string());
			}

			google::protobuf::RepeatedPtrField<std::string>
			node_ids(nodes.begin(), nodes.end());

			op::GetDataRequest req;
			req.mutable_uuids()->Swap(&node_ids);
			for (auto& ign : ignored)
			{
				if (auto id = iosvc_->lookup_id(ign))
				{
					req.add_ignored(*id);
				}
			}

			completions.push_back(client->get_data(cq_, req,
				[this, peer_id](op::NodeData& res)
				{
					auto uuid = res.uuid();
					auto ref = static_cast<iDistrRef*>(
						iosvc_->must_lookup_node(uuid).get());
					ref->update_data(res.data().data(), res.version());
				}));
		}
		// wait for completion before evaluating in local
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
		// locally evaluate
		teq::TravEvaluator eval(device, ignored);
		teq::multi_visit(eval, targets);
	}

	/// Return map of reachable src tensors mapped to reachable dest ids
	teq::TensMapT<types::StrUSetT> reachable (
		error::ErrptrT& err,
		const teq::TensSetT& srcs,
		const types::StrUSetT& dests)
	{
		teq::TensMapT<std::string> local_targets;
		for (auto dest : dests)
		{
			error::ErrptrT e;
			if (auto local = iosvc_->lookup_node(e, dest, false))
			{
				local_targets.emplace(local.get(), dest);
			}
		}
		teq::TensMapT<types::StrUSetT> reached;
		types::StrUMapT<types::StrUSetT> remotes;
		types::StrUMapT<teq::TensSetT> refsrcs;
		for (auto src : srcs)
		{
			if (estd::has(reach_cache_, src))
			{
				auto& reachs = reach_cache_[src];
				types::StrUSetT potentials;
				std::set_intersection(reachs.begin(), reachs.end(),
					dests.begin(), dests.end(),
					std::inserter(potentials, potentials.end()));
				if (potentials.size() > 0)
				{
					reached.emplace(src, potentials);
					continue;
				}
			}
			teq::LambdaVisit vis(
				[&](teq::iLeaf& leaf)
				{
					std::string id;
					if (estd::get(id, local_targets, &leaf))
					{
						reached[src].emplace(id);
						reach_cache_[src].emplace(id);
					}
				},
				[&](teq::iTraveler& trav, teq::iFunctor& func)
				{
					std::string id;
					if (estd::has(reached, src) ||
						estd::get(id, local_targets, &func))
					{
						reached[src].emplace(id);
						reach_cache_[src].emplace(id);
						return;
					}
					teq::multi_visit(trav, func.get_dependencies());
				});
			src->accept(vis);
			if (false == estd::has(reached, src))
			{
				auto refs = reachable_refs(teq::TensT{src});
				separate_by_server(remotes, refs);
				for (auto ref : refs)
				{
					refsrcs[ref->node_id()].emplace(src);
				}
			}
		}
		if (remotes.size() > 0)
		{
			std::list<egrpc::ErrPromiseptrT> completions;
			for (auto& remote : remotes)
			{
				auto peer_id = remote.first;
				auto subsrcs = remote.second;
				auto client = get_client(err, peer_id);
				if (nullptr != err)
				{
					return {};
				}
				op::ListReachableRequest req;
				google::protobuf::RepeatedPtrField<std::string>
				src_uuids(subsrcs.begin(), subsrcs.end());
				google::protobuf::RepeatedPtrField<std::string>
				dest_uuids(dests.begin(), dests.end());
				req.mutable_srcs()->Swap(&src_uuids);
				req.mutable_dests()->Swap(&dest_uuids);
				completions.push_back(
					client->list_reachable(cq_, req,
					[&](op::ListReachableResponse& res)
					{
						types::StrUMapT<types::StrUSetT> reachables;
						auto& res_src = res.srcs();
						for (auto& ress : res_src)
						{
							auto& rec = ress.second.reachables();
							reachables.emplace(ress.first,
								types::StrUSetT(rec.begin(), rec.end()));
						}
						for (auto reachable : reachables)
						{
							auto& src = reachable.first;
							auto& targs = reachable.second;
							auto& rsrcs = refsrcs[src];
							for (auto rsrc : rsrcs)
							{
								reached[rsrc].insert(targs.begin(), targs.end());
							}
						}
					}));
			}
			while (false == completions.empty())
			{
				auto done = completions.front()->get_future();
				wait_on_future(done);
				completions.pop_front();
			}
		}
		return reached;
	}

	teq::OwnMapT derive (
		teq::GradMapT& grads,
		const teq::TensptrSetT& roots,
		const BackpropMeta& metas,
		const teq::iDerivativeFuncs& builder)
	{
		if (roots.empty())
		{
			return {};
		}

		auto& targets = metas.targets_;

		// only look at reachable refs
		auto refs = reachable_refs(roots);
		teq::TensSetT refset(refs.begin(), refs.end());

		teq::TensptrSetT locals;
		for (auto root : roots)
		{
			if (false == estd::has(refset, root.get()))
			{
				locals.emplace(root);
			}
		}

		types::StrUSetT targids;
		for (auto target : targets)
		{
			// targets that are not exposed can't be referenced remotely
			if (auto id = iosvc_->lookup_id(target))
			{
				targids.emplace(*id);
			}
		}
		error::ErrptrT err = nullptr;
		auto reachs = estd::map_keyset(reachable(err, refset, targids));
		if (nullptr != err)
		{
			global::fatal(err->to_string());
		}
		DRefSetT reachrefs;
		std::transform(reachs.begin(), reachs.end(),
			std::inserter(reachrefs, reachrefs.end()),
			[](teq::iTensor* tens) { return static_cast<iDistrRef*>(tens); });

		// populate grads by local gradients
		if (locals.size() > 0)
		{
			// filter target references by target reachability
			auto local_targets = targets;
			local_targets.insert(reachs.begin(), reachs.end());
			teq::partial_derive(grads, locals, local_targets, builder);
		}

		// then make remote calls
		types::StrUMapT<types::StrUSetT> remotes;
		separate_by_server(remotes, reachrefs);
		if (remotes.size() > 0)
		{
			std::list<egrpc::ErrPromiseptrT> completions;
			std::string rootid = "";//*lookup_id(metas.root_.get());
			for (auto remote : remotes)
			{
				types::StrUMapT<teq::TensptrT> pgrads;
				for (auto pid : remote.second)
				{
					auto parent = iosvc_->must_lookup_node(pid);
					teq::TensptrT tens;
					teq::TensptrsT tgrads;
					if (estd::get(tgrads, grads, parent.get()) && tgrads.size() > 0)
					{
						tens = tgrads.size() == 1 ?
							tgrads.front() : builder.add(tgrads);
						grads.erase(parent.get()); // clear local derivatives to avoid duplicates
					}
					else
					{
						tens = builder.get_const_zero(parent->shape());
					}
					pgrads.emplace(pid, tens);
				}
				if (pgrads.size() > 0)
				{
					// make remote call
					auto peer_id = remote.first;

					error::ErrptrT err = nullptr;
					auto client = get_client(err, peer_id);
					if (nullptr != err)
					{
						global::fatal(err->to_string());
					}

					op::CreateDeriveRequest req;
					req.set_root(rootid);
					for (auto target : targids)
					{
						req.add_targets(target);
					}
					auto pbpgrads = req.mutable_root_grads();
					for (auto& rgrad : pgrads)
					{
						auto& pbmeta = (*pbpgrads)[rgrad.first];
						auto tens = rgrad.second;
						auto uuid = iosvc_->expose_node(tens);
						tens_to_node_meta(pbmeta, get_peer_id(), uuid, tens);
					}
					completions.push_back(
						client->create_derive(cq_, req,
						[&](op::CreateDeriveResponse& res)
						{
							auto& target_grads = res.grads();
							types::StrUMapT<std::string> tgrads(
								target_grads.begin(), target_grads.end());

							for (auto tgrad : tgrads)
							{
								std::string tid = tgrad.first;
								std::string gid = tgrad.second;
								grads[iosvc_->must_lookup_node(tid).get()].
									push_back(iosvc_->must_lookup_node(gid));
							}
						}));
				}
				while (false == completions.empty())
				{
					auto done = completions.front()->get_future();
					wait_on_future(done);
					completions.pop_front();
				}
			}
		}

		teq::OwnMapT out;
		for (auto target : targets)
		{
			teq::TensptrT tens;
			teq::TensptrsT tgrads;
			if (estd::get(tgrads, grads, target) && tgrads.size() > 0)
			{
				tens = tgrads.size() == 1 ?
					tgrads.front() : builder.add(tgrads);
			}
			else
			{
				tens = builder.get_const_zero(target->shape());
			}
			out.emplace(target, tens);
		}
		return out;
	}

	void register_service (grpc::ServerBuilder& builder) override
	{
		builder.RegisterService(&service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		// GetData
		auto gdata_logger = std::make_shared<global::FormatLogger>(
			&global::get_logger(), fmts::sprintf("[server %s:GetData] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerStreamCall<op::GetDataRequest,
			op::NodeData,DataStatesT>(gdata_logger,
			[this](grpc::ServerContext* ctx, op::GetDataRequest* req,
				grpc::ServerAsyncWriter<op::NodeData>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestGetData(ctx, req, writer, cq, ccq, tag);
			},
			[this](DataStatesT& states, const op::GetDataRequest& req)
			{
				return this->startup_get_data(states, req);
			},
			process_get_data, &cq);

		// ListReachable
		auto lreachable_logger = std::make_shared<global::FormatLogger>(
			&global::get_logger(), fmts::sprintf("[server %s:ListReachable] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerCall<op::ListReachableRequest,
			op::ListReachableResponse>(lreachable_logger,
			[this](grpc::ServerContext* ctx, op::ListReachableRequest* req,
				grpc::ServerAsyncResponseWriter<op::ListReachableResponse>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestListReachable(ctx, req, writer, cq, ccq, tag);
			},
			[this](
				const op::ListReachableRequest& req,
				op::ListReachableResponse& res)
			{
				return this->list_reachable(req, res);
			}, &cq);

		// Derive
		auto cderive_logger = std::make_shared<global::FormatLogger>(
			&global::get_logger(), fmts::sprintf("[server %s:CreateDerive] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerCall<op::CreateDeriveRequest,
			op::CreateDeriveResponse>(cderive_logger,
			[this](grpc::ServerContext* ctx, op::CreateDeriveRequest* req,
				grpc::ServerAsyncResponseWriter<op::CreateDeriveResponse>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestCreateDerive(ctx, req, writer, cq, ccq, tag);
			},
			[this](
				const op::CreateDeriveRequest& req,
				op::CreateDeriveResponse& res)
			{
				return this->create_derive(req, res);
			}, &cq);
	}

private:
	grpc::Status startup_get_data (
		DataStatesT& states, const op::GetDataRequest& req)
	{
		auto& uuids = req.uuids();
		auto& ig_ids = req.ignored();
		teq::TensSetT targets;
		teq::TensSetT ignored;
		for (const std::string& uuid : uuids)
		{
			error::ErrptrT err = nullptr;
			auto tens = iosvc_->lookup_node(err, uuid, false).get();
			_ERR_CHECK(err, grpc::NOT_FOUND,
				fmts::sprintf("%s:GetData", get_peer_id().c_str()).c_str());
			targets.emplace(tens);
			states.emplace(uuid, tens);
		}
		for (const std::string& uuid : ig_ids)
		{
			error::ErrptrT err = nullptr;
			auto tens = iosvc_->lookup_node(err, uuid, false).get();
			_ERR_CHECK(err, grpc::NOT_FOUND,
				fmts::sprintf("%s:GetData", get_peer_id().c_str()).c_str());
			ignored.emplace(tens);
		}
		eigen::Device device(std::numeric_limits<size_t>::max());
		evaluate(device, targets, ignored);
		return grpc::Status::OK;
	}

	grpc::Status list_reachable (
		const op::ListReachableRequest& req,
		op::ListReachableResponse& res)
	{
		auto alias = fmts::sprintf("%s:ListReachable",
			get_peer_id().c_str());
		error::ErrptrT err = nullptr;
		auto& dests = req.dests();
		teq::TensSetT srcs;
		for (auto src : req.srcs())
		{
			auto local = iosvc_->lookup_node(err, src, false);
			_ERR_CHECK(err, grpc::NOT_FOUND, alias.c_str());
			srcs.emplace(local.get());
		}
		auto reached = reachable(err, srcs,
			types::StrUSetT(dests.begin(), dests.end()));
		_ERR_CHECK(err, grpc::NOT_FOUND, alias.c_str());
		auto payload = res.mutable_srcs();
		for (auto& r : reached)
		{
			std::string id = *iosvc_->lookup_id(r.first);
			google::protobuf::RepeatedPtrField<std::string>
			targs(r.second.begin(), r.second.end());
			(*payload)[id].mutable_reachables()->Swap(&targs);
		}
		return grpc::Status::OK;
	}

	grpc::Status create_derive (
		const op::CreateDeriveRequest& req,
		op::CreateDeriveResponse& res)
	{
		auto alias = fmts::sprintf("%s:CreateDerive", get_peer_id().c_str());
		auto& rgrads = req.root_grads();
		auto& targids = req.targets();
		// // cache based on rootid (todo)
		// std::string rootid = req.root();

		error::ErrptrT err = nullptr;
		teq::GradMapT grads;
		teq::TensptrSetT parents;
		teq::TensSetT targets;
		teq::TensMapT<std::string> targmap;
		// populate grads and parents from request
		for (auto& reqpairs : rgrads)
		{
			auto local_id = reqpairs.first;
			auto gradmeta = reqpairs.second;
			auto local = iosvc_->lookup_node(err, local_id, false);
			_ERR_CHECK(err, grpc::NOT_FOUND, alias.c_str());
			auto grad = node_meta_to_ref(gradmeta);
			iosvc_->expose_node(grad);
			grads[local.get()].push_back(grad);
			parents.emplace(local);
		}
		for (auto targid : targids)
		{
			auto target = iosvc_->lookup_node(err, targid);
			_ERR_CHECK(err, grpc::NOT_FOUND, alias.c_str());
			targets.emplace(target.get());
			targmap.emplace(target.get(), targid);
		}
		// run local derivation
		teq::OwnMapT tgrads;
		if (parents.size() > 0)
		{
			teq::iDerivativeFuncs* builder = nullptr;
			auto dtype = (*parents.begin())->get_meta().type_code();
			TYPE_LOOKUP(_MAKE_DERFUNC, dtype);
			tgrads = derive(grads, parents, BackpropMeta{targets}, *builder);
			delete builder;
		}
		// populate response grads
		auto resgrads = res.mutable_grads();
		for (auto targ : tgrads)
		{
			auto targid = targmap[targ.first];
			auto grad = targ.second;
			iosvc_->expose_node(grad);
			std::string gid = *iosvc_->lookup_id(grad.get());
			resgrads->insert(google::protobuf::MapPair<
				std::string,std::string>{targid, gid});
		}
		return grpc::Status::OK;
	}

	DistrIOService* iosvc_;

	op::DistrOperation::AsyncService service_;

	// todo: move to data obj
	teq::TensMapT<types::StrUSetT> reach_cache_;
};

#undef _MAKE_DERFUNC

#undef _ERR_CHECK

error::ErrptrT register_opsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

DistrOpService& get_opsvc (iDistrManager& manager);

}

#endif // DISTRIB_OP_SERVICE_HPP
