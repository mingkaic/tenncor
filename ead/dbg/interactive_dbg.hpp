#include "ead/constant.hpp"
#include "ead/functor.hpp"

#include "ead/dbg/interactive_client.hpp"

#ifndef EAD_DBG_HPP
#define EAD_DBG_HPP

namespace ead
{

static const std::string default_server = "localhost:50051";

const std::string default_label = "normal_graph";

const std::string continue_msg = "continue (y/N)? >";

const std::string invalid_break_signal_msg =
    "accepts y/n/yes/no (case insensitive) as response";

struct Painter : public ade::iTraveler
{
    Painter (GraphCanvasT& canvas, std::string label, bool overwrite) :
        canvas_(&canvas), label_(label), overwrite_(overwrite) {}

    /// Implementation of iTraveler
    void visit (ade::iLeaf* leaf) override
    {
        if (visited_.end() == visited_.find(leaf) &&
            (canvas_->end() == canvas_->find(leaf) || overwrite_))
        {
            visited_.emplace(leaf);
            canvas_->emplace(leaf, NodeMetadata{false, label_, std::weak_ptr<ade::iTensor>()});
        }
    }

    /// Implementation of iTraveler
    void visit (ade::iFunctor* func) override
    {
        if (visited_.end() == visited_.find(func) &&
            (canvas_->end() == canvas_->find(func) || overwrite_))
        {
            visited_.emplace(func);
            canvas_->emplace(func, NodeMetadata{true, label_, std::weak_ptr<ade::iTensor>()});
            auto children = func->get_children();
            for (auto& child : children)
            {
                auto ctens = child.get_tensor();
                ctens->accept(*this);
                (*canvas_)[ctens.get()].ownership_ = ctens;
            }
        }
    }

    std::unordered_set<ade::iTensor*> visited_;

    GraphCanvasT* canvas_;

    std::string label_;

    bool overwrite_;
};

template <typename T>
struct InteractiveDebugger
{
    void track_label (NodeptrT<T>& node, std::string label = default_label)
    {
        Painter painter(graph_canvas_, label, false);
        auto root = node->get_tensor();
        root->accept(painter);
        graph_canvas_[root.get()].ownership_ = root;
    }

    void track_overwrite_label (NodeptrT<T>& node, std::string label)
    {
        Painter painter(graph_canvas_, label, true);
        auto root = node->get_tensor();
        root->accept(painter);
        graph_canvas_[root.get()].ownership_ = root;
    }

    // update display then wait signal before continue (single-thread blocker)
    void set_break (void)
    {
        // cleanup canvas
        for (auto it = graph_canvas_.begin(), et = graph_canvas_.end(); it != et;)
        {
            auto& owner = it->second.ownership_;
            if (owner.expired())
            {
                it = graph_canvas_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        // update display
        GraphClient client(default_server);
        client.send(graph_canvas_);

        // handle signal
        // todo: make more elegant and flexible
        std::string break_signal;
        while (!std::cin.eof())
        {
            std::cout << continue_msg;
            std::getline(std::cin, break_signal);
            // parse break_signal
            std::transform(break_signal.begin(), break_signal.end(),
                break_signal.begin(), [](char c) { return std::tolower(c); });
            if (break_signal == "y" || break_signal == "yes")
            {
                return;
            }
            else if (break_signal != "n" && break_signal != "no")
            {
                logs::warn(invalid_break_signal_msg);
            }
            break_signal = "";
        }
    }

    GraphCanvasT graph_canvas_;
};

}

#endif // EAD_DBG_HPP
