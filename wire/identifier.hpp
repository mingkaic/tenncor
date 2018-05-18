/*!
 *
 *  identifier.hpp
 *  wire
 *
 *  Purpose:
 *  node proxy to enforce id and labeling functionality,
 *  and safely destroy
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/inode.hpp"
#include "mold/iobserver.hpp"

#include "wire/graph.hpp"

#pragma once
#ifndef WIRE_IDENTIFIER_HPP
#define WIRE_IDENTIFIER_HPP

namespace wire
{
    
class Identifier : public iObserver
{
public:
    Identifier (Graph* graph, iNode* arg) :
        iObserver({arg}), graph_(graph)
    {
        id_ = graph->associate(arg, this);
    }

    virtual ~Identifier (void)
    {
        graph->disassociate(id_);
    }

    Identifier (const Identifier& other) :
        iObserver(other)
    {
        graph_->disassociate(id_);
        graph_ = other.graph_;
        // assert false == args_.empty()
        graph_->associate(args_[0], this);
    }

    Identifier (Identifier&& other) :
        iObserver(std::move(other))
    {
        graph_->disassociate(id_);
        graph_ = std::move(other.graph_);
        // assert false == args_.empty()
        graph_->associate(args_[0], this);
    }

    void initialize (void) override {} // todo: add functionality

    void update (void) override {} // todo: add functionality

private:
    Graph* graph_;

    std::string label_;

    std::string id_;
};

}

#endif /* WIRE_IDENTIFIER_HPP */
