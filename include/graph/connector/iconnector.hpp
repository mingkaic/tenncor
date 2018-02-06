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
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
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
	virtual ~iconnector (void);

	//! clone function
	iconnector* clone (void) const;

	//! move function
	iconnector* move (void);

	//! declare copy assignment to enforce proper g_man_ copy over
	virtual iconnector& operator = (const iconnector& other);

	//! declare move assignment to enforce proper g_man_ copy over
	virtual iconnector& operator = (iconnector&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> IDENTIFICATION <<<<<<

	//! get unique label with arguments
	virtual std::string get_name (void) const;

	//! get the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	virtual size_t get_depth (void) const;

	// >>>>>> ICONNECTOR SPECIAL <<<<<<

	//! check if other connector is in the same graph as this
	bool is_same_graph (const iconnector* other) const;

	//! check if connector n is a potential descendent of this node
	//! will return false negatives if nodes are in a pipeline of a non-variable leaf
	virtual bool potential_descendent (const iconnector* n) const;

	//! get all observerables
	std::vector<inode*> get_arguments (void) const;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	// >>>>>> FORWARD & BACKWARD DATA <<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	// >>>>>> ICONNECTOR SPECIAL <<<<<<

	//! freeze or unfreeze the current node
	//! freeze prevents this from updating temporarily instead update is queued to g_man_
	void freeze_status (bool freeze);

protected:
	//! Set dependencies
	iconnector (std::vector<inode*> dependencies, std::string label);

	//! Declare copy constructor to enforce proper g_man_ copy over
	iconnector (const iconnector& other);

	//! Declare move constructor to enforce proper g_man_ copy over
	iconnector (iconnector&& other);

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

	// >>>>>> MANAGE GRAPH INFO <<<<<<

	//! Update g_man_ by updating all argument variables
	virtual void update_graph (std::vector<iconnector*> args);

	//! graph meta_data/manager
	graph_manager* g_man_ = nullptr;

private:
	void copy_helper (const iconnector& other);

	void move_helper (iconnector&& other);

	//! the distance between this node and the furthest dependent leaf (maximum spanning tree height)
	size_t depth_ = 0;
};

}

#endif /* TENNCOR_ICONNECTOR_HPP */
