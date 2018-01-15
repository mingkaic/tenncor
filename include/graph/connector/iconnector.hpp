/*!
 *
 *  iconnector.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph connector interface
 *  manages graph information
 *
 *  Created by Mingkai Chen on 2016-12-01.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <queue>

#include "include/graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_ICONNECTOR_HPP
#define TENNCOR_ICONNECTOR_HPP

namespace nnet
{

//! backward transfer function, get gradient nodes; F: Nf -> Nb
using BACK_MAP = std::function<varptr(std::vector<std::pair<inode*,inode*>>)>;

using NODE_MAN = std::function<inode*(inode*)>;

//! jacobian transfer function
using JTRANSFER = std::function<inode*(inode*,std::vector<inode*>,std::vector<inode*>)>;

//! calculate output shape from argument shapes
using SHAPER = std::function<tensorshape(std::vector<tensorshape>)>;

class iconnector : public inode, public iobserver
{
public:
	//! iconnector summary
	struct conn_summary
	{
		conn_summary (std::string id, SHAPER shaper,
			TRANSFER_FUNC<double> forward, BACK_MAP back) :
				id_(id), shaper_(shaper), Nf_(forward), ginit_(back) {}

		std::string id_;
		SHAPER shaper_;
		TRANSFER_FUNC<double> Nf_;
		BACK_MAP ginit_;

		std::vector<std::string> arg_ids_;
	};

	using summary_series = std::vector<typename iconnector::conn_summary>;

	virtual ~iconnector (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	iconnector* clone (void) const;

	//! move function
	iconnector* move (void);

	//! declare copy assignment to enforce proper g_man_ copy over
	virtual iconnector& operator = (const iconnector& other);

	//! declare move assignment to enforce proper g_man_ copy over
	virtual iconnector& operator = (iconnector&& other);

	// >>>> IDENTIFICATION <<<<
	//! get unique label with arguments
	virtual std::string get_name (void) const;

	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const;

	// >>>> OBSERVER & OBSERVABLE INFO <<<<
	//! get all observerables
	virtual std::vector<inode*> get_arguments (void) const;

	//! get the number of observables
	virtual size_t n_arguments (void) const;

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! grab a temporary value traversing top-down
	virtual void temporary_eval (const iconnector* target, inode*& out) const = 0;

	//! get forward passing value, (pull data if necessary)
	virtual const tensor<double>* eval (void);

	// >>>> GRAPH STATUS <<<<
	//! check if other connector is in the same graph as this
	bool is_same_graph (const iconnector* other) const;

	//! check if connector n is a potential descendent of this node
	//! will return false negatives if nodes are in a pipeline of a non-variable leaf
	virtual bool potential_descendent (const iconnector* n) const;

	// >>>> NODE STATUS <<<<
	//! add jacobian to the front of the list mapped by leaves
	void set_jacobian_front (JTRANSFER jac, std::vector<variable*> leaves);

	//! add jacobian to the back of the list mapped by leaves
	void set_jacobian_back (JTRANSFER jac, std::vector<variable*> leaves);

	//! freeze or unfreeze the current node
	//! freeze prevents this from updating temporarily instead update is queued to g_man_
	void freeze_status (bool freeze);

protected:
	//! list of jacobian transfer function
	//! to be executed on resulting root node
	//! execution order: top-down
	struct jlist
	{
		std::string uid_ = nnutils::uuid(this);
		std::list<std::pair<JTRANSFER, inode*>> list_;
		bool terminal_ = false;
	};

	//! graph info shareable between connectors
	struct graph_manager
	{
		static graph_manager* get (iconnector* source, iconnector* user = nullptr)
		{
			assert(source);
			graph_manager*& gn = source->g_man_;
			if (nullptr == gn) gn = new graph_manager();
			if (nullptr == user) user = source;
			gn->users_.emplace(user);
			return gn;
		}

		graph_manager (const graph_manager&) = delete;

		graph_manager (graph_manager&&) = delete;

		void suicide (iconnector* user)
		{
			users_.erase(user);
			if (users_.empty()) delete this;
		}

		void consume (graph_manager* other)
		{
			if (this == other) return;
			while (false == other->updates_.empty())
			{
				updates_.push(other->updates_.top());
				other->updates_.pop();
			}
			update_map_.insert(other->update_map_.begin(), other->update_map_.end());
			std::unordered_set<iconnector*> otherusers = other->users_;
			for (iconnector* ouser : otherusers)
			{
				other->suicide(ouser);
				ouser->g_man_ = this;
			}
			users_.insert(otherusers.begin(), otherusers.end());
		}

		void add_update (iconnector* dependent, std::function<void(void)> update)
		{
			// assert dependent is in users_
			if (update_map_.end() == update_map_.find(dependent))
			{
				updates_.push(dependent);
				update_map_[dependent] = update;
			}
		}

		void update (void)
		{
			// todo: add multithreading
			while (false == updates_.empty())
			{
				iconnector* iconn = updates_.top();
				auto updater = update_map_[iconn];
				updates_.pop();
				updater();
				iconn->notify(UPDATE);
			}
			update_map_.clear();
		}

		bool freeze_ = false;

	private:
		struct small_leafset
		{
			bool operator() (const iconnector* c1, const iconnector* c2) const
			{
				return c1->get_depth()> c2->get_depth();
			}
		};

		std::priority_queue<iconnector*,std::vector<iconnector*>,small_leafset> updates_;

		std::unordered_map<iconnector*,std::function<void(void)>> update_map_;

		std::unordered_set<iconnector*> users_;
		
		graph_manager (void) {}
		
		~graph_manager (void) {}
	};

	// >>>> CONSTRUCTORS <<<<
	//! Set dependencies
	iconnector (std::vector<inode*> dependencies, std::string label);

	//! Declare copy constructor to enforce proper g_man_ copy over
	iconnector (const iconnector& other);

	//! Declare move constructor to enforce proper g_man_ copy over
	iconnector (iconnector&& other);

	// >>>> MANAGE GRAPH INFO <<<<
	//! Update g_man_ by updating all argument variables
	virtual void update_graph (std::vector<iconnector*> args);

	varptr jacobian_call (varptr out, variable* leaf) const;

	//! specialized operator: jacobian operators for each variable,
	//! executed in derive
	std::unordered_map<variable*,jlist> jacobians_;

	//! graph meta_data/manager
	graph_manager* g_man_ = nullptr;

private:
	void copy_helper (const iconnector& other);

	void move_helper (iconnector&& other);

	void jacobian_correction (const inode* other);

	//! the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	size_t depth_ = 0;
};

}

#endif /* TENNCOR_ICONNECTOR_HPP */
