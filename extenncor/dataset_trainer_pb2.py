# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: extenncor/dataset_trainer.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='extenncor/dataset_trainer.proto',
  package='dataset',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1f\x65xtenncor/dataset_trainer.proto\x12\x07\x64\x61taset\"=\n\x08OnxDSEnv\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x61taset_idx\x18\x02 \x01(\x05\x12\x0e\n\x06oxfile\x18\x03 \x01(\tb\x06proto3'
)




_ONXDSENV = _descriptor.Descriptor(
  name='OnxDSEnv',
  full_name='dataset.OnxDSEnv',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='dataset.OnxDSEnv.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset_idx', full_name='dataset.OnxDSEnv.dataset_idx', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='oxfile', full_name='dataset.OnxDSEnv.oxfile', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=105,
)

DESCRIPTOR.message_types_by_name['OnxDSEnv'] = _ONXDSENV
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OnxDSEnv = _reflection.GeneratedProtocolMessageType('OnxDSEnv', (_message.Message,), {
  'DESCRIPTOR' : _ONXDSENV,
  '__module__' : 'extenncor.dataset_trainer_pb2'
  # @@protoc_insertion_point(class_scope:dataset.OnxDSEnv)
  })
_sym_db.RegisterMessage(OnxDSEnv)


# @@protoc_insertion_point(module_scope)