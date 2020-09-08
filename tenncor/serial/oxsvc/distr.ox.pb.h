// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tenncor/serial/oxsvc/distr.ox.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3011000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3011000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/map.h>  // IWYU pragma: export
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
#include "internal/onnx/onnx.pb.h"
#include "tenncor/distr/iosvc/distr.io.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[5]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto;
namespace distr {
namespace ox {
class GetSaveGraphRequest;
class GetSaveGraphRequestDefaultTypeInternal;
extern GetSaveGraphRequestDefaultTypeInternal _GetSaveGraphRequest_default_instance_;
class GetSaveGraphResponse;
class GetSaveGraphResponseDefaultTypeInternal;
extern GetSaveGraphResponseDefaultTypeInternal _GetSaveGraphResponse_default_instance_;
class PostLoadGraphRequest;
class PostLoadGraphRequestDefaultTypeInternal;
extern PostLoadGraphRequestDefaultTypeInternal _PostLoadGraphRequest_default_instance_;
class PostLoadGraphRequest_TopographyEntry_DoNotUse;
class PostLoadGraphRequest_TopographyEntry_DoNotUseDefaultTypeInternal;
extern PostLoadGraphRequest_TopographyEntry_DoNotUseDefaultTypeInternal _PostLoadGraphRequest_TopographyEntry_DoNotUse_default_instance_;
class PostLoadGraphResponse;
class PostLoadGraphResponseDefaultTypeInternal;
extern PostLoadGraphResponseDefaultTypeInternal _PostLoadGraphResponse_default_instance_;
}  // namespace ox
}  // namespace distr
PROTOBUF_NAMESPACE_OPEN
template<> ::distr::ox::GetSaveGraphRequest* Arena::CreateMaybeMessage<::distr::ox::GetSaveGraphRequest>(Arena*);
template<> ::distr::ox::GetSaveGraphResponse* Arena::CreateMaybeMessage<::distr::ox::GetSaveGraphResponse>(Arena*);
template<> ::distr::ox::PostLoadGraphRequest* Arena::CreateMaybeMessage<::distr::ox::PostLoadGraphRequest>(Arena*);
template<> ::distr::ox::PostLoadGraphRequest_TopographyEntry_DoNotUse* Arena::CreateMaybeMessage<::distr::ox::PostLoadGraphRequest_TopographyEntry_DoNotUse>(Arena*);
template<> ::distr::ox::PostLoadGraphResponse* Arena::CreateMaybeMessage<::distr::ox::PostLoadGraphResponse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace distr {
namespace ox {

// ===================================================================

class GetSaveGraphRequest :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:distr.ox.GetSaveGraphRequest) */ {
 public:
  GetSaveGraphRequest();
  virtual ~GetSaveGraphRequest();

  GetSaveGraphRequest(const GetSaveGraphRequest& from);
  GetSaveGraphRequest(GetSaveGraphRequest&& from) noexcept
    : GetSaveGraphRequest() {
    *this = ::std::move(from);
  }

  inline GetSaveGraphRequest& operator=(const GetSaveGraphRequest& from) {
    CopyFrom(from);
    return *this;
  }
  inline GetSaveGraphRequest& operator=(GetSaveGraphRequest&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const GetSaveGraphRequest& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const GetSaveGraphRequest* internal_default_instance() {
    return reinterpret_cast<const GetSaveGraphRequest*>(
               &_GetSaveGraphRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(GetSaveGraphRequest& a, GetSaveGraphRequest& b) {
    a.Swap(&b);
  }
  inline void Swap(GetSaveGraphRequest* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GetSaveGraphRequest* New() const final {
    return CreateMaybeMessage<GetSaveGraphRequest>(nullptr);
  }

  GetSaveGraphRequest* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GetSaveGraphRequest>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const GetSaveGraphRequest& from);
  void MergeFrom(const GetSaveGraphRequest& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(GetSaveGraphRequest* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "distr.ox.GetSaveGraphRequest";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto);
    return ::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kUuidsFieldNumber = 1,
  };
  // repeated string uuids = 1;
  int uuids_size() const;
  private:
  int _internal_uuids_size() const;
  public:
  void clear_uuids();
  const std::string& uuids(int index) const;
  std::string* mutable_uuids(int index);
  void set_uuids(int index, const std::string& value);
  void set_uuids(int index, std::string&& value);
  void set_uuids(int index, const char* value);
  void set_uuids(int index, const char* value, size_t size);
  std::string* add_uuids();
  void add_uuids(const std::string& value);
  void add_uuids(std::string&& value);
  void add_uuids(const char* value);
  void add_uuids(const char* value, size_t size);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>& uuids() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>* mutable_uuids();
  private:
  const std::string& _internal_uuids(int index) const;
  std::string* _internal_add_uuids();
  public:

  // @@protoc_insertion_point(class_scope:distr.ox.GetSaveGraphRequest)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string> uuids_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto;
};
// -------------------------------------------------------------------

class GetSaveGraphResponse :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:distr.ox.GetSaveGraphResponse) */ {
 public:
  GetSaveGraphResponse();
  virtual ~GetSaveGraphResponse();

  GetSaveGraphResponse(const GetSaveGraphResponse& from);
  GetSaveGraphResponse(GetSaveGraphResponse&& from) noexcept
    : GetSaveGraphResponse() {
    *this = ::std::move(from);
  }

  inline GetSaveGraphResponse& operator=(const GetSaveGraphResponse& from) {
    CopyFrom(from);
    return *this;
  }
  inline GetSaveGraphResponse& operator=(GetSaveGraphResponse&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const GetSaveGraphResponse& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const GetSaveGraphResponse* internal_default_instance() {
    return reinterpret_cast<const GetSaveGraphResponse*>(
               &_GetSaveGraphResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(GetSaveGraphResponse& a, GetSaveGraphResponse& b) {
    a.Swap(&b);
  }
  inline void Swap(GetSaveGraphResponse* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GetSaveGraphResponse* New() const final {
    return CreateMaybeMessage<GetSaveGraphResponse>(nullptr);
  }

  GetSaveGraphResponse* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GetSaveGraphResponse>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const GetSaveGraphResponse& from);
  void MergeFrom(const GetSaveGraphResponse& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(GetSaveGraphResponse* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "distr.ox.GetSaveGraphResponse";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto);
    return ::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kGraphFieldNumber = 1,
  };
  // .onnx.GraphProto graph = 1;
  bool has_graph() const;
  private:
  bool _internal_has_graph() const;
  public:
  void clear_graph();
  const ::onnx::GraphProto& graph() const;
  ::onnx::GraphProto* release_graph();
  ::onnx::GraphProto* mutable_graph();
  void set_allocated_graph(::onnx::GraphProto* graph);
  private:
  const ::onnx::GraphProto& _internal_graph() const;
  ::onnx::GraphProto* _internal_mutable_graph();
  public:

  // @@protoc_insertion_point(class_scope:distr.ox.GetSaveGraphResponse)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::onnx::GraphProto* graph_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto;
};
// -------------------------------------------------------------------

class PostLoadGraphRequest_TopographyEntry_DoNotUse : public ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<PostLoadGraphRequest_TopographyEntry_DoNotUse, 
    std::string, std::string,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    0 > {
public:
  typedef ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<PostLoadGraphRequest_TopographyEntry_DoNotUse, 
    std::string, std::string,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    0 > SuperType;
  PostLoadGraphRequest_TopographyEntry_DoNotUse();
  PostLoadGraphRequest_TopographyEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  void MergeFrom(const PostLoadGraphRequest_TopographyEntry_DoNotUse& other);
  static const PostLoadGraphRequest_TopographyEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const PostLoadGraphRequest_TopographyEntry_DoNotUse*>(&_PostLoadGraphRequest_TopographyEntry_DoNotUse_default_instance_); }
  static bool ValidateKey(std::string* s) {
    return ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(s->data(), static_cast<int>(s->size()), ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE, "distr.ox.PostLoadGraphRequest.TopographyEntry.key");
 }
  static bool ValidateValue(std::string* s) {
    return ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(s->data(), static_cast<int>(s->size()), ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE, "distr.ox.PostLoadGraphRequest.TopographyEntry.value");
 }
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& other) final;
  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto);
    return ::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto.file_level_metadata[2];
  }

  public:
};

// -------------------------------------------------------------------

class PostLoadGraphRequest :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:distr.ox.PostLoadGraphRequest) */ {
 public:
  PostLoadGraphRequest();
  virtual ~PostLoadGraphRequest();

  PostLoadGraphRequest(const PostLoadGraphRequest& from);
  PostLoadGraphRequest(PostLoadGraphRequest&& from) noexcept
    : PostLoadGraphRequest() {
    *this = ::std::move(from);
  }

  inline PostLoadGraphRequest& operator=(const PostLoadGraphRequest& from) {
    CopyFrom(from);
    return *this;
  }
  inline PostLoadGraphRequest& operator=(PostLoadGraphRequest&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const PostLoadGraphRequest& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const PostLoadGraphRequest* internal_default_instance() {
    return reinterpret_cast<const PostLoadGraphRequest*>(
               &_PostLoadGraphRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    3;

  friend void swap(PostLoadGraphRequest& a, PostLoadGraphRequest& b) {
    a.Swap(&b);
  }
  inline void Swap(PostLoadGraphRequest* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline PostLoadGraphRequest* New() const final {
    return CreateMaybeMessage<PostLoadGraphRequest>(nullptr);
  }

  PostLoadGraphRequest* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<PostLoadGraphRequest>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const PostLoadGraphRequest& from);
  void MergeFrom(const PostLoadGraphRequest& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(PostLoadGraphRequest* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "distr.ox.PostLoadGraphRequest";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto);
    return ::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------


  // accessors -------------------------------------------------------

  enum : int {
    kTopographyFieldNumber = 2,
    kGraphFieldNumber = 1,
  };
  // map<string, string> topography = 2;
  int topography_size() const;
  private:
  int _internal_topography_size() const;
  public:
  void clear_topography();
  private:
  const ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >&
      _internal_topography() const;
  ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >*
      _internal_mutable_topography();
  public:
  const ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >&
      topography() const;
  ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >*
      mutable_topography();

  // .onnx.GraphProto graph = 1;
  bool has_graph() const;
  private:
  bool _internal_has_graph() const;
  public:
  void clear_graph();
  const ::onnx::GraphProto& graph() const;
  ::onnx::GraphProto* release_graph();
  ::onnx::GraphProto* mutable_graph();
  void set_allocated_graph(::onnx::GraphProto* graph);
  private:
  const ::onnx::GraphProto& _internal_graph() const;
  ::onnx::GraphProto* _internal_mutable_graph();
  public:

  // @@protoc_insertion_point(class_scope:distr.ox.PostLoadGraphRequest)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::MapField<
      PostLoadGraphRequest_TopographyEntry_DoNotUse,
      std::string, std::string,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
      0 > topography_;
  ::onnx::GraphProto* graph_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto;
};
// -------------------------------------------------------------------

class PostLoadGraphResponse :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:distr.ox.PostLoadGraphResponse) */ {
 public:
  PostLoadGraphResponse();
  virtual ~PostLoadGraphResponse();

  PostLoadGraphResponse(const PostLoadGraphResponse& from);
  PostLoadGraphResponse(PostLoadGraphResponse&& from) noexcept
    : PostLoadGraphResponse() {
    *this = ::std::move(from);
  }

  inline PostLoadGraphResponse& operator=(const PostLoadGraphResponse& from) {
    CopyFrom(from);
    return *this;
  }
  inline PostLoadGraphResponse& operator=(PostLoadGraphResponse&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const PostLoadGraphResponse& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const PostLoadGraphResponse* internal_default_instance() {
    return reinterpret_cast<const PostLoadGraphResponse*>(
               &_PostLoadGraphResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    4;

  friend void swap(PostLoadGraphResponse& a, PostLoadGraphResponse& b) {
    a.Swap(&b);
  }
  inline void Swap(PostLoadGraphResponse* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline PostLoadGraphResponse* New() const final {
    return CreateMaybeMessage<PostLoadGraphResponse>(nullptr);
  }

  PostLoadGraphResponse* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<PostLoadGraphResponse>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const PostLoadGraphResponse& from);
  void MergeFrom(const PostLoadGraphResponse& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(PostLoadGraphResponse* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "distr.ox.PostLoadGraphResponse";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto);
    return ::descriptor_table_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kValuesFieldNumber = 1,
  };
  // repeated .distr.io.NodeMeta values = 1;
  int values_size() const;
  private:
  int _internal_values_size() const;
  public:
  void clear_values();
  ::distr::io::NodeMeta* mutable_values(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::distr::io::NodeMeta >*
      mutable_values();
  private:
  const ::distr::io::NodeMeta& _internal_values(int index) const;
  ::distr::io::NodeMeta* _internal_add_values();
  public:
  const ::distr::io::NodeMeta& values(int index) const;
  ::distr::io::NodeMeta* add_values();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::distr::io::NodeMeta >&
      values() const;

  // @@protoc_insertion_point(class_scope:distr.ox.PostLoadGraphResponse)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::distr::io::NodeMeta > values_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// GetSaveGraphRequest

// repeated string uuids = 1;
inline int GetSaveGraphRequest::_internal_uuids_size() const {
  return uuids_.size();
}
inline int GetSaveGraphRequest::uuids_size() const {
  return _internal_uuids_size();
}
inline void GetSaveGraphRequest::clear_uuids() {
  uuids_.Clear();
}
inline std::string* GetSaveGraphRequest::add_uuids() {
  // @@protoc_insertion_point(field_add_mutable:distr.ox.GetSaveGraphRequest.uuids)
  return _internal_add_uuids();
}
inline const std::string& GetSaveGraphRequest::_internal_uuids(int index) const {
  return uuids_.Get(index);
}
inline const std::string& GetSaveGraphRequest::uuids(int index) const {
  // @@protoc_insertion_point(field_get:distr.ox.GetSaveGraphRequest.uuids)
  return _internal_uuids(index);
}
inline std::string* GetSaveGraphRequest::mutable_uuids(int index) {
  // @@protoc_insertion_point(field_mutable:distr.ox.GetSaveGraphRequest.uuids)
  return uuids_.Mutable(index);
}
inline void GetSaveGraphRequest::set_uuids(int index, const std::string& value) {
  // @@protoc_insertion_point(field_set:distr.ox.GetSaveGraphRequest.uuids)
  uuids_.Mutable(index)->assign(value);
}
inline void GetSaveGraphRequest::set_uuids(int index, std::string&& value) {
  // @@protoc_insertion_point(field_set:distr.ox.GetSaveGraphRequest.uuids)
  uuids_.Mutable(index)->assign(std::move(value));
}
inline void GetSaveGraphRequest::set_uuids(int index, const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  uuids_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:distr.ox.GetSaveGraphRequest.uuids)
}
inline void GetSaveGraphRequest::set_uuids(int index, const char* value, size_t size) {
  uuids_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:distr.ox.GetSaveGraphRequest.uuids)
}
inline std::string* GetSaveGraphRequest::_internal_add_uuids() {
  return uuids_.Add();
}
inline void GetSaveGraphRequest::add_uuids(const std::string& value) {
  uuids_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:distr.ox.GetSaveGraphRequest.uuids)
}
inline void GetSaveGraphRequest::add_uuids(std::string&& value) {
  uuids_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:distr.ox.GetSaveGraphRequest.uuids)
}
inline void GetSaveGraphRequest::add_uuids(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  uuids_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:distr.ox.GetSaveGraphRequest.uuids)
}
inline void GetSaveGraphRequest::add_uuids(const char* value, size_t size) {
  uuids_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:distr.ox.GetSaveGraphRequest.uuids)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>&
GetSaveGraphRequest::uuids() const {
  // @@protoc_insertion_point(field_list:distr.ox.GetSaveGraphRequest.uuids)
  return uuids_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>*
GetSaveGraphRequest::mutable_uuids() {
  // @@protoc_insertion_point(field_mutable_list:distr.ox.GetSaveGraphRequest.uuids)
  return &uuids_;
}

// -------------------------------------------------------------------

// GetSaveGraphResponse

// .onnx.GraphProto graph = 1;
inline bool GetSaveGraphResponse::_internal_has_graph() const {
  return this != internal_default_instance() && graph_ != nullptr;
}
inline bool GetSaveGraphResponse::has_graph() const {
  return _internal_has_graph();
}
inline const ::onnx::GraphProto& GetSaveGraphResponse::_internal_graph() const {
  const ::onnx::GraphProto* p = graph_;
  return p != nullptr ? *p : *reinterpret_cast<const ::onnx::GraphProto*>(
      &::onnx::_GraphProto_default_instance_);
}
inline const ::onnx::GraphProto& GetSaveGraphResponse::graph() const {
  // @@protoc_insertion_point(field_get:distr.ox.GetSaveGraphResponse.graph)
  return _internal_graph();
}
inline ::onnx::GraphProto* GetSaveGraphResponse::release_graph() {
  // @@protoc_insertion_point(field_release:distr.ox.GetSaveGraphResponse.graph)
  
  ::onnx::GraphProto* temp = graph_;
  graph_ = nullptr;
  return temp;
}
inline ::onnx::GraphProto* GetSaveGraphResponse::_internal_mutable_graph() {
  
  if (graph_ == nullptr) {
    auto* p = CreateMaybeMessage<::onnx::GraphProto>(GetArenaNoVirtual());
    graph_ = p;
  }
  return graph_;
}
inline ::onnx::GraphProto* GetSaveGraphResponse::mutable_graph() {
  // @@protoc_insertion_point(field_mutable:distr.ox.GetSaveGraphResponse.graph)
  return _internal_mutable_graph();
}
inline void GetSaveGraphResponse::set_allocated_graph(::onnx::GraphProto* graph) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(graph_);
  }
  if (graph) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      graph = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, graph, submessage_arena);
    }
    
  } else {
    
  }
  graph_ = graph;
  // @@protoc_insertion_point(field_set_allocated:distr.ox.GetSaveGraphResponse.graph)
}

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// PostLoadGraphRequest

// .onnx.GraphProto graph = 1;
inline bool PostLoadGraphRequest::_internal_has_graph() const {
  return this != internal_default_instance() && graph_ != nullptr;
}
inline bool PostLoadGraphRequest::has_graph() const {
  return _internal_has_graph();
}
inline const ::onnx::GraphProto& PostLoadGraphRequest::_internal_graph() const {
  const ::onnx::GraphProto* p = graph_;
  return p != nullptr ? *p : *reinterpret_cast<const ::onnx::GraphProto*>(
      &::onnx::_GraphProto_default_instance_);
}
inline const ::onnx::GraphProto& PostLoadGraphRequest::graph() const {
  // @@protoc_insertion_point(field_get:distr.ox.PostLoadGraphRequest.graph)
  return _internal_graph();
}
inline ::onnx::GraphProto* PostLoadGraphRequest::release_graph() {
  // @@protoc_insertion_point(field_release:distr.ox.PostLoadGraphRequest.graph)
  
  ::onnx::GraphProto* temp = graph_;
  graph_ = nullptr;
  return temp;
}
inline ::onnx::GraphProto* PostLoadGraphRequest::_internal_mutable_graph() {
  
  if (graph_ == nullptr) {
    auto* p = CreateMaybeMessage<::onnx::GraphProto>(GetArenaNoVirtual());
    graph_ = p;
  }
  return graph_;
}
inline ::onnx::GraphProto* PostLoadGraphRequest::mutable_graph() {
  // @@protoc_insertion_point(field_mutable:distr.ox.PostLoadGraphRequest.graph)
  return _internal_mutable_graph();
}
inline void PostLoadGraphRequest::set_allocated_graph(::onnx::GraphProto* graph) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(graph_);
  }
  if (graph) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      graph = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, graph, submessage_arena);
    }
    
  } else {
    
  }
  graph_ = graph;
  // @@protoc_insertion_point(field_set_allocated:distr.ox.PostLoadGraphRequest.graph)
}

// map<string, string> topography = 2;
inline int PostLoadGraphRequest::_internal_topography_size() const {
  return topography_.size();
}
inline int PostLoadGraphRequest::topography_size() const {
  return _internal_topography_size();
}
inline void PostLoadGraphRequest::clear_topography() {
  topography_.Clear();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >&
PostLoadGraphRequest::_internal_topography() const {
  return topography_.GetMap();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >&
PostLoadGraphRequest::topography() const {
  // @@protoc_insertion_point(field_map:distr.ox.PostLoadGraphRequest.topography)
  return _internal_topography();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >*
PostLoadGraphRequest::_internal_mutable_topography() {
  return topography_.MutableMap();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< std::string, std::string >*
PostLoadGraphRequest::mutable_topography() {
  // @@protoc_insertion_point(field_mutable_map:distr.ox.PostLoadGraphRequest.topography)
  return _internal_mutable_topography();
}

// -------------------------------------------------------------------

// PostLoadGraphResponse

// repeated .distr.io.NodeMeta values = 1;
inline int PostLoadGraphResponse::_internal_values_size() const {
  return values_.size();
}
inline int PostLoadGraphResponse::values_size() const {
  return _internal_values_size();
}
inline ::distr::io::NodeMeta* PostLoadGraphResponse::mutable_values(int index) {
  // @@protoc_insertion_point(field_mutable:distr.ox.PostLoadGraphResponse.values)
  return values_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::distr::io::NodeMeta >*
PostLoadGraphResponse::mutable_values() {
  // @@protoc_insertion_point(field_mutable_list:distr.ox.PostLoadGraphResponse.values)
  return &values_;
}
inline const ::distr::io::NodeMeta& PostLoadGraphResponse::_internal_values(int index) const {
  return values_.Get(index);
}
inline const ::distr::io::NodeMeta& PostLoadGraphResponse::values(int index) const {
  // @@protoc_insertion_point(field_get:distr.ox.PostLoadGraphResponse.values)
  return _internal_values(index);
}
inline ::distr::io::NodeMeta* PostLoadGraphResponse::_internal_add_values() {
  return values_.Add();
}
inline ::distr::io::NodeMeta* PostLoadGraphResponse::add_values() {
  // @@protoc_insertion_point(field_add:distr.ox.PostLoadGraphResponse.values)
  return _internal_add_values();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::distr::io::NodeMeta >&
PostLoadGraphResponse::values() const {
  // @@protoc_insertion_point(field_list:distr.ox.PostLoadGraphResponse.values)
  return values_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace ox
}  // namespace distr

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tenncor_2fserial_2foxsvc_2fdistr_2eox_2eproto
