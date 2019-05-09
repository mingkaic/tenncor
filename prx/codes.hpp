#include "ead/generated/codes.hpp"

#ifndef PRX_CODES_HPP
#define PRX_CODES_HPP

namespace prx
{

// todo: add control layer (for dropout)
enum GRAPH_LAYER
{
    BAD_LAYER = age::_N_GENERATED_OPCODES,
    FULL_CONN,
    CONV_LAYER,
    SOFT_MAX,
    _N_GRAPH_LAYERS,
};

enum GRAPH
{
    BAD_GRAPH = _N_GRAPH_LAYERS,
    MPERCEPTRON, // mlp
    DQUALITY, // dqn
    DBELIEF, // dbn
    CONV_NET, // cnn
    _N_GRAPH,
};

std::string name_layer (GRAPH_LAYER code);

GRAPH_LAYER get_layer (std::string name);

std::string name_graph (GRAPH code);

GRAPH get_graph (std::string name);

}

#endif // PRX_CODES_HPP
