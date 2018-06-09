//
//  error.cpp
//  kiln
//

#include "kiln/error.hpp"

#include "ioutil/stream.hpp"

#ifdef KILN_ERROR_HPP

namespace kiln
{

DuplicateNodeIDError::DuplicateNodeIDError (
	std::string gid, UID node_id) :
	std::runtime_error(ioutil::Stream() <<
		"duplicate id " << node_id <<
		" found in graph " << gid) {}

MissingNodeError::MissingNodeError (
	std::string gid, UID node_id) :
	std::runtime_error(ioutil::Stream() <<
		"node " << node_id <<
		" missing from graph " << gid) {}

}

#endif
