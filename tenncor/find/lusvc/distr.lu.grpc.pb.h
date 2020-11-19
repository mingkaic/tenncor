// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: tenncor/find/lusvc/distr.lu.proto
#ifndef GRPC_tenncor_2ffind_2flusvc_2fdistr_2elu_2eproto__INCLUDED
#define GRPC_tenncor_2ffind_2flusvc_2fdistr_2elu_2eproto__INCLUDED

#include "tenncor/find/lusvc/distr.lu.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace grpc_impl {
class CompletionQueue;
class ServerCompletionQueue;
class ServerContext;
}  // namespace grpc_impl

namespace grpc {
namespace experimental {
template <typename RequestT, typename ResponseT>
class MessageAllocator;
}  // namespace experimental
}  // namespace grpc

namespace distr {
namespace lu {

class DistrLookup final {
 public:
  static constexpr char const* service_full_name() {
    return "distr.lu.DistrLookup";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status ListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::distr::lu::ListNodesResponse* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::distr::lu::ListNodesResponse>> AsyncListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::distr::lu::ListNodesResponse>>(AsyncListNodesRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::distr::lu::ListNodesResponse>> PrepareAsyncListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::distr::lu::ListNodesResponse>>(PrepareAsyncListNodesRaw(context, request, cq));
    }
    class experimental_async_interface {
     public:
      virtual ~experimental_async_interface() {}
      virtual void ListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest* request, ::distr::lu::ListNodesResponse* response, std::function<void(::grpc::Status)>) = 0;
      virtual void ListNodes(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::distr::lu::ListNodesResponse* response, std::function<void(::grpc::Status)>) = 0;
      virtual void ListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest* request, ::distr::lu::ListNodesResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      virtual void ListNodes(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::distr::lu::ListNodesResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
    };
    virtual class experimental_async_interface* experimental_async() { return nullptr; }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::distr::lu::ListNodesResponse>* AsyncListNodesRaw(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::distr::lu::ListNodesResponse>* PrepareAsyncListNodesRaw(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status ListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::distr::lu::ListNodesResponse* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::distr::lu::ListNodesResponse>> AsyncListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::distr::lu::ListNodesResponse>>(AsyncListNodesRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::distr::lu::ListNodesResponse>> PrepareAsyncListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::distr::lu::ListNodesResponse>>(PrepareAsyncListNodesRaw(context, request, cq));
    }
    class experimental_async final :
      public StubInterface::experimental_async_interface {
     public:
      void ListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest* request, ::distr::lu::ListNodesResponse* response, std::function<void(::grpc::Status)>) override;
      void ListNodes(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::distr::lu::ListNodesResponse* response, std::function<void(::grpc::Status)>) override;
      void ListNodes(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest* request, ::distr::lu::ListNodesResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      void ListNodes(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::distr::lu::ListNodesResponse* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
     private:
      friend class Stub;
      explicit experimental_async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class experimental_async_interface* experimental_async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class experimental_async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::distr::lu::ListNodesResponse>* AsyncListNodesRaw(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::distr::lu::ListNodesResponse>* PrepareAsyncListNodesRaw(::grpc::ClientContext* context, const ::distr::lu::ListNodesRequest& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_ListNodes_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status ListNodes(::grpc::ServerContext* context, const ::distr::lu::ListNodesRequest* request, ::distr::lu::ListNodesResponse* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_ListNodes : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_ListNodes() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_ListNodes() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ListNodes(::grpc::ServerContext* /*context*/, const ::distr::lu::ListNodesRequest* /*request*/, ::distr::lu::ListNodesResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestListNodes(::grpc::ServerContext* context, ::distr::lu::ListNodesRequest* request, ::grpc::ServerAsyncResponseWriter< ::distr::lu::ListNodesResponse>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_ListNodes<Service > AsyncService;
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_ListNodes : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithCallbackMethod_ListNodes() {
      ::grpc::Service::experimental().MarkMethodCallback(0,
        new ::grpc_impl::internal::CallbackUnaryHandler< ::distr::lu::ListNodesRequest, ::distr::lu::ListNodesResponse>(
          [this](::grpc::ServerContext* context,
                 const ::distr::lu::ListNodesRequest* request,
                 ::distr::lu::ListNodesResponse* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   return this->ListNodes(context, request, response, controller);
                 }));
    }
    void SetMessageAllocatorFor_ListNodes(
        ::grpc::experimental::MessageAllocator< ::distr::lu::ListNodesRequest, ::distr::lu::ListNodesResponse>* allocator) {
      static_cast<::grpc_impl::internal::CallbackUnaryHandler< ::distr::lu::ListNodesRequest, ::distr::lu::ListNodesResponse>*>(
          ::grpc::Service::experimental().GetHandler(0))
              ->SetMessageAllocator(allocator);
    }
    ~ExperimentalWithCallbackMethod_ListNodes() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ListNodes(::grpc::ServerContext* /*context*/, const ::distr::lu::ListNodesRequest* /*request*/, ::distr::lu::ListNodesResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void ListNodes(::grpc::ServerContext* /*context*/, const ::distr::lu::ListNodesRequest* /*request*/, ::distr::lu::ListNodesResponse* /*response*/, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  typedef ExperimentalWithCallbackMethod_ListNodes<Service > ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_ListNodes : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_ListNodes() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_ListNodes() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ListNodes(::grpc::ServerContext* /*context*/, const ::distr::lu::ListNodesRequest* /*request*/, ::distr::lu::ListNodesResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_ListNodes : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_ListNodes() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_ListNodes() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ListNodes(::grpc::ServerContext* /*context*/, const ::distr::lu::ListNodesRequest* /*request*/, ::distr::lu::ListNodesResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestListNodes(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_ListNodes : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithRawCallbackMethod_ListNodes() {
      ::grpc::Service::experimental().MarkMethodRawCallback(0,
        new ::grpc_impl::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
          [this](::grpc::ServerContext* context,
                 const ::grpc::ByteBuffer* request,
                 ::grpc::ByteBuffer* response,
                 ::grpc::experimental::ServerCallbackRpcController* controller) {
                   this->ListNodes(context, request, response, controller);
                 }));
    }
    ~ExperimentalWithRawCallbackMethod_ListNodes() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status ListNodes(::grpc::ServerContext* /*context*/, const ::distr::lu::ListNodesRequest* /*request*/, ::distr::lu::ListNodesResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    virtual void ListNodes(::grpc::ServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/, ::grpc::experimental::ServerCallbackRpcController* controller) { controller->Finish(::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "")); }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_ListNodes : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_ListNodes() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< ::distr::lu::ListNodesRequest, ::distr::lu::ListNodesResponse>(std::bind(&WithStreamedUnaryMethod_ListNodes<BaseClass>::StreamedListNodes, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_ListNodes() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status ListNodes(::grpc::ServerContext* /*context*/, const ::distr::lu::ListNodesRequest* /*request*/, ::distr::lu::ListNodesResponse* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedListNodes(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::distr::lu::ListNodesRequest,::distr::lu::ListNodesResponse>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_ListNodes<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_ListNodes<Service > StreamedService;
};

}  // namespace lu
}  // namespace distr


#endif  // GRPC_tenncor_2ffind_2flusvc_2fdistr_2elu_2eproto__INCLUDED