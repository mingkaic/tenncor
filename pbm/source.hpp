///
/// source.hpp
/// pbm
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "adhoc/llo/node.hpp"

#include "pbm/graph.pb.h"

#ifndef PBM_SOURCE_HPP
#define PBM_SOURCE_HPP

/// Marshal llo::iSource to tenncor::Source
void save_data (tenncor::Source* out, llo::iSource* in);

/// Unmarshal tenncor::Source as DataNode containing context of source
llo::DataNode load_source (const tenncor::Source& source);

#endif // PBM_SOURCE_HPP
