//
//  error.cpp
//  wire
//

#include "wire/error.hpp"

#include "ioutil/stream.hpp"

#ifdef WIRE_ERROR_HPP

namespace wire
{

DuplicateNodeIDError::DuplicateNodeIDError (
	std::string gid, std::string node_id) :
	std::runtime_error(ioutil::Stream() <<
		"duplicate id " << node_id <<
		" found in graph " << gid) {}

MissingNodeError::MissingNodeError (
	std::string gid, std::string node_id) :
	std::runtime_error(ioutil::Stream() <<
		"node " << node_id <<
		" missing from graph " << gid) {}

}

#endif
