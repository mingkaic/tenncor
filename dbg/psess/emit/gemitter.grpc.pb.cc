// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: dbg/psess/emit/gemitter.proto

#include "dbg/psess/emit/gemitter.pb.h"
#include "dbg/psess/emit/gemitter.grpc.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/channel_interface.h>
#include <grpcpp/impl/codegen/client_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/rpc_service_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/sync_stream.h>
namespace gemitter {

static const char* GraphEmitter_method_names[] = {
  "/gemitter.GraphEmitter/HealthCheck",
  "/gemitter.GraphEmitter/CreateModel",
  "/gemitter.GraphEmitter/UpdateNodeData",
  "/gemitter.GraphEmitter/DeleteModel",
};

std::unique_ptr< GraphEmitter::Stub> GraphEmitter::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr< GraphEmitter::Stub> stub(new GraphEmitter::Stub(channel));
  return stub;
}

GraphEmitter::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_HealthCheck_(GraphEmitter_method_names[0], ::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_CreateModel_(GraphEmitter_method_names[1], ::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_UpdateNodeData_(GraphEmitter_method_names[2], ::grpc::internal::RpcMethod::CLIENT_STREAMING, channel)
  , rpcmethod_DeleteModel_(GraphEmitter_method_names[3], ::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status GraphEmitter::Stub::HealthCheck(::grpc::ClientContext* context, const ::gemitter::Empty& request, ::gemitter::Empty* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_HealthCheck_, context, request, response);
}

void GraphEmitter::Stub::experimental_async::HealthCheck(::grpc::ClientContext* context, const ::gemitter::Empty* request, ::gemitter::Empty* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_HealthCheck_, context, request, response, std::move(f));
}

void GraphEmitter::Stub::experimental_async::HealthCheck(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::gemitter::Empty* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_HealthCheck_, context, request, response, std::move(f));
}

void GraphEmitter::Stub::experimental_async::HealthCheck(::grpc::ClientContext* context, const ::gemitter::Empty* request, ::gemitter::Empty* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_HealthCheck_, context, request, response, reactor);
}

void GraphEmitter::Stub::experimental_async::HealthCheck(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::gemitter::Empty* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_HealthCheck_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::gemitter::Empty>* GraphEmitter::Stub::AsyncHealthCheckRaw(::grpc::ClientContext* context, const ::gemitter::Empty& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::gemitter::Empty>::Create(channel_.get(), cq, rpcmethod_HealthCheck_, context, request, true);
}

::grpc::ClientAsyncResponseReader< ::gemitter::Empty>* GraphEmitter::Stub::PrepareAsyncHealthCheckRaw(::grpc::ClientContext* context, const ::gemitter::Empty& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::gemitter::Empty>::Create(channel_.get(), cq, rpcmethod_HealthCheck_, context, request, false);
}

::grpc::Status GraphEmitter::Stub::CreateModel(::grpc::ClientContext* context, const ::gemitter::CreateModelRequest& request, ::gemitter::CreateModelResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_CreateModel_, context, request, response);
}

void GraphEmitter::Stub::experimental_async::CreateModel(::grpc::ClientContext* context, const ::gemitter::CreateModelRequest* request, ::gemitter::CreateModelResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_CreateModel_, context, request, response, std::move(f));
}

void GraphEmitter::Stub::experimental_async::CreateModel(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::gemitter::CreateModelResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_CreateModel_, context, request, response, std::move(f));
}

void GraphEmitter::Stub::experimental_async::CreateModel(::grpc::ClientContext* context, const ::gemitter::CreateModelRequest* request, ::gemitter::CreateModelResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_CreateModel_, context, request, response, reactor);
}

void GraphEmitter::Stub::experimental_async::CreateModel(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::gemitter::CreateModelResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_CreateModel_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::gemitter::CreateModelResponse>* GraphEmitter::Stub::AsyncCreateModelRaw(::grpc::ClientContext* context, const ::gemitter::CreateModelRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::gemitter::CreateModelResponse>::Create(channel_.get(), cq, rpcmethod_CreateModel_, context, request, true);
}

::grpc::ClientAsyncResponseReader< ::gemitter::CreateModelResponse>* GraphEmitter::Stub::PrepareAsyncCreateModelRaw(::grpc::ClientContext* context, const ::gemitter::CreateModelRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::gemitter::CreateModelResponse>::Create(channel_.get(), cq, rpcmethod_CreateModel_, context, request, false);
}

::grpc::ClientWriter< ::gemitter::UpdateNodeDataRequest>* GraphEmitter::Stub::UpdateNodeDataRaw(::grpc::ClientContext* context, ::gemitter::UpdateNodeDataResponse* response) {
  return ::grpc_impl::internal::ClientWriterFactory< ::gemitter::UpdateNodeDataRequest>::Create(channel_.get(), rpcmethod_UpdateNodeData_, context, response);
}

void GraphEmitter::Stub::experimental_async::UpdateNodeData(::grpc::ClientContext* context, ::gemitter::UpdateNodeDataResponse* response, ::grpc::experimental::ClientWriteReactor< ::gemitter::UpdateNodeDataRequest>* reactor) {
  ::grpc_impl::internal::ClientCallbackWriterFactory< ::gemitter::UpdateNodeDataRequest>::Create(stub_->channel_.get(), stub_->rpcmethod_UpdateNodeData_, context, response, reactor);
}

::grpc::ClientAsyncWriter< ::gemitter::UpdateNodeDataRequest>* GraphEmitter::Stub::AsyncUpdateNodeDataRaw(::grpc::ClientContext* context, ::gemitter::UpdateNodeDataResponse* response, ::grpc::CompletionQueue* cq, void* tag) {
  return ::grpc_impl::internal::ClientAsyncWriterFactory< ::gemitter::UpdateNodeDataRequest>::Create(channel_.get(), cq, rpcmethod_UpdateNodeData_, context, response, true, tag);
}

::grpc::ClientAsyncWriter< ::gemitter::UpdateNodeDataRequest>* GraphEmitter::Stub::PrepareAsyncUpdateNodeDataRaw(::grpc::ClientContext* context, ::gemitter::UpdateNodeDataResponse* response, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncWriterFactory< ::gemitter::UpdateNodeDataRequest>::Create(channel_.get(), cq, rpcmethod_UpdateNodeData_, context, response, false, nullptr);
}

::grpc::Status GraphEmitter::Stub::DeleteModel(::grpc::ClientContext* context, const ::gemitter::DeleteModelRequest& request, ::gemitter::DeleteModelResponse* response) {
  return ::grpc::internal::BlockingUnaryCall(channel_.get(), rpcmethod_DeleteModel_, context, request, response);
}

void GraphEmitter::Stub::experimental_async::DeleteModel(::grpc::ClientContext* context, const ::gemitter::DeleteModelRequest* request, ::gemitter::DeleteModelResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_DeleteModel_, context, request, response, std::move(f));
}

void GraphEmitter::Stub::experimental_async::DeleteModel(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::gemitter::DeleteModelResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc_impl::internal::CallbackUnaryCall(stub_->channel_.get(), stub_->rpcmethod_DeleteModel_, context, request, response, std::move(f));
}

void GraphEmitter::Stub::experimental_async::DeleteModel(::grpc::ClientContext* context, const ::gemitter::DeleteModelRequest* request, ::gemitter::DeleteModelResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_DeleteModel_, context, request, response, reactor);
}

void GraphEmitter::Stub::experimental_async::DeleteModel(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::gemitter::DeleteModelResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) {
  ::grpc_impl::internal::ClientCallbackUnaryFactory::Create(stub_->channel_.get(), stub_->rpcmethod_DeleteModel_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::gemitter::DeleteModelResponse>* GraphEmitter::Stub::AsyncDeleteModelRaw(::grpc::ClientContext* context, const ::gemitter::DeleteModelRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::gemitter::DeleteModelResponse>::Create(channel_.get(), cq, rpcmethod_DeleteModel_, context, request, true);
}

::grpc::ClientAsyncResponseReader< ::gemitter::DeleteModelResponse>* GraphEmitter::Stub::PrepareAsyncDeleteModelRaw(::grpc::ClientContext* context, const ::gemitter::DeleteModelRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc_impl::internal::ClientAsyncResponseReaderFactory< ::gemitter::DeleteModelResponse>::Create(channel_.get(), cq, rpcmethod_DeleteModel_, context, request, false);
}

GraphEmitter::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      GraphEmitter_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< GraphEmitter::Service, ::gemitter::Empty, ::gemitter::Empty>(
          std::mem_fn(&GraphEmitter::Service::HealthCheck), this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      GraphEmitter_method_names[1],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< GraphEmitter::Service, ::gemitter::CreateModelRequest, ::gemitter::CreateModelResponse>(
          std::mem_fn(&GraphEmitter::Service::CreateModel), this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      GraphEmitter_method_names[2],
      ::grpc::internal::RpcMethod::CLIENT_STREAMING,
      new ::grpc::internal::ClientStreamingHandler< GraphEmitter::Service, ::gemitter::UpdateNodeDataRequest, ::gemitter::UpdateNodeDataResponse>(
          std::mem_fn(&GraphEmitter::Service::UpdateNodeData), this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      GraphEmitter_method_names[3],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< GraphEmitter::Service, ::gemitter::DeleteModelRequest, ::gemitter::DeleteModelResponse>(
          std::mem_fn(&GraphEmitter::Service::DeleteModel), this)));
}

GraphEmitter::Service::~Service() {
}

::grpc::Status GraphEmitter::Service::HealthCheck(::grpc::ServerContext* context, const ::gemitter::Empty* request, ::gemitter::Empty* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status GraphEmitter::Service::CreateModel(::grpc::ServerContext* context, const ::gemitter::CreateModelRequest* request, ::gemitter::CreateModelResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status GraphEmitter::Service::UpdateNodeData(::grpc::ServerContext* context, ::grpc::ServerReader< ::gemitter::UpdateNodeDataRequest>* reader, ::gemitter::UpdateNodeDataResponse* response) {
  (void) context;
  (void) reader;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status GraphEmitter::Service::DeleteModel(::grpc::ServerContext* context, const ::gemitter::DeleteModelRequest* request, ::gemitter::DeleteModelResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace gemitter
