
// #include "egrpc/egrpc.hpp"

// #include "tenncor/eteq/eteq.hpp"

// #include "tenncor/distrib/imanager.hpp"
// #include "tenncor/eteq/distr_svc/serialize/client.hpp"

// #ifndef DISTRIB_SERIALIZE_SERVICE_HPP
// #define DISTRIB_SERIALIZE_SERVICE_HPP

// namespace distr
// {

// const std::string serializesvc_key = "distr_serializesvc";

// struct DistrSerializeService final : public PeerService<DistrSerializeCli>
// {
// 	DistrSerializeService (const PeerServiceConfig& cfg) :
// 		PeerService<DistrSerializeCli>(cfg) {}

// 	template <typename TS> // todo: use concept tensptr_range
// 	void save_graph (onnx::GraphProto& pb_graph, const TS& roots,
// 		const onnx::iMarshFuncs& marshaler, const onnx::TensIdT& identified = {})
// 	{
// 		auto refs = distr::reachable_refs(roots);
// 		types::StrUMapT<types::StrUSetT> remotes;
// 		distr::separate_by_server(remotes, refs);

// 		for (auto& remote : remotes)
// 		{
// 			//
// 		}

// 		onnx::OnnxMarshaler marshal(pb_graph, identified, marshaler);
// 		// add references to tens to delay serialization of references
// 		teq::multi_visit(marshal, roots);

// 		std::vector<const teq::iTensor*> rtens(
// 			marshal.roots_.begin(), marshal.roots_.end());
// 		std::sort(rtens.begin(), rtens.end(),
// 			[&marshal](const teq::iTensor* a, const teq::iTensor* b)
// 			{
// 				return marshal.tens_.at(a) < marshal.tens_.at(b);
// 			});
// 		for (const teq::iTensor* root : rtens)
// 		{
// 			onnx::ValueInfoProto* pb_output = pb_graph.add_output();
// 			pb_output->set_name(marshal.tens_.at(root));
// 			marshal_io(*pb_output, root->shape());
// 		}
// 	}

// 	void register_service (grpc::ServerBuilder& builder) override
// 	{
// 		builder.RegisterService(&service_);
// 	}

// 	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
// 	{
// 		// GetData
// 		auto gdata_logger = std::make_shared<global::FormatLogger>(
// 			global::get_logger(), fmts::sprintf("[server %s:GetData] ",
// 				get_peer_id().c_str()));
// 		new egrpc::AsyncServerStreamCall<serialize::GetDataRequest,
// 			serialize::NodeData,DataStatesT>(gdata_logger,
// 			[this](grpc::ServerContext* ctx, serialize::GetDataRequest* req,
// 				grpc::ServerAsyncWriter<serialize::NodeData>* writer,
// 				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
// 				void* tag)
// 			{
// 				this->service_.RequestGetData(ctx, req, writer, cq, ccq, tag);
// 			},
// 			[this](DataStatesT& states, const serialize::GetDataRequest& req)
// 			{
// 				return this->startup_get_data(states, req);
// 			},
// 			process_get_data, &cq);

// 		// ListReachable
// 		auto lreachable_logger = std::make_shared<global::FormatLogger>(
// 			global::get_logger(), fmts::sprintf("[server %s:ListReachable] ",
// 				get_peer_id().c_str()));
// 		new egrpc::AsyncServerCall<serialize::ListReachableRequest,
// 			serialize::ListReachableResponse>(lreachable_logger,
// 			[this](grpc::ServerContext* ctx, serialize::ListReachableRequest* req,
// 				grpc::ServerAsyncResponseWriter<serialize::ListReachableResponse>* writer,
// 				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
// 				void* tag)
// 			{
// 				this->service_.RequestListReachable(ctx, req, writer, cq, ccq, tag);
// 			},
// 			[this](
// 				const serialize::ListReachableRequest& req,
// 				serialize::ListReachableResponse& res)
// 			{
// 				return this->list_reachable(req, res);
// 			}, &cq);

// 		// Derive
// 		auto cderive_logger = std::make_shared<global::FormatLogger>(
// 			global::get_logger(), fmts::sprintf("[server %s:CreateDerive] ",
// 				get_peer_id().c_str()));
// 		new egrpc::AsyncServerCall<serialize::CreateDeriveRequest,
// 			serialize::CreateDeriveResponse>(cderive_logger,
// 			[this](grpc::ServerContext* ctx, serialize::CreateDeriveRequest* req,
// 				grpc::ServerAsyncResponseWriter<serialize::CreateDeriveResponse>* writer,
// 				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
// 				void* tag)
// 			{
// 				this->service_.RequestCreateDerive(ctx, req, writer, cq, ccq, tag);
// 			},
// 			[this](
// 				const serialize::CreateDeriveRequest& req,
// 				serialize::CreateDeriveResponse& res)
// 			{
// 				return this->create_derive(req, res);
// 			}, &cq);
// 	}

// private:
// 	serialize::DistrSerializeeration::AsyncService service_;
// };

// error::ErrptrT register_serializesvc (estd::ConfigMap<>& svcs,
// 	const PeerServiceConfig& cfg);

// DistrSerializeService& get_serializesvc (iDistrManager& manager);

// }

// #endif // DISTRIB_SERIALIZE_SERVICE_HPP
