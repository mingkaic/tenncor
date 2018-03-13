/*!
 *
 *  iobserver.hpp
 *  cnnet
 *
 *  Purpose:
 *  observer interface is notified
 *  by subjects when changes occur
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/react/subject.hpp"

#pragma once
#ifndef TENNCOR_IOBSERVER_HPP
#define TENNCOR_IOBSERVER_HPP

#include <vector>
#include <functional>
#include <algorithm>

namespace nnet
{

class iobserver
{
public:
	virtual ~iobserver (void);

	//! declare copy assignment to copy over dependencies
	virtual iobserver& operator = (const iobserver& other);

	//! declare move assignment to move over dependencies
	virtual iobserver& operator = (iobserver&& other);



	// >>>>>>>>>>>> ACCESSOR <<<<<<<<<<<<

	//! determine whether this observes sub
	bool has_subject (subject* sub) const;

	bool is_recordable (void) const;

protected:
	//! default constructor
	iobserver (bool recordable = true);

	//! subscribe to subjects on construction (mostly non-mutable observers)
	iobserver (std::vector<subject*> dependencies, bool recordable = true);

	//! copy over dependencies
	iobserver (const iobserver& other);

	//! move over dependencies
	iobserver (iobserver&& other);



	// >>>>>>>>>>>> MUTATOR <<<<<<<<<<<<

	//! update observer value according to subject
	//! publicly available to allow explicit updates
	virtual void update (void) = 0;

	// >>>>>> KILL CONDITION <<<<<<

	//! smart destruction: call when any observer is broken
	virtual void death_on_broken (void) = 0;

	void add_ondeath_dependent (subject* dep);

	void remove_ondeath_dependent (subject* dep);

	// >>>>>> DEPENDENCY MUTATORS <<<<<<

	//! subscribe: add dependency
	void add_dependency (subject* dep);

	//! unsubscribe: remove dependency
	void remove_dependency (size_t idx);

	//! replace dependency
	void replace_dependency (subject* dep, size_t i);

	// >>>>>> NOTIFICATION MESSAGE MANAGER <<<<<<

	//! update observer value with notification
	virtual void update (notification msg, std::unordered_set<size_t> indices);

	//! order of subject matters;
	//! observer-subject relation is non-unique
	std::vector<subject*> dependencies_;

private:
	//! copy helper function
	void copy_helper (const iobserver& other);

	//! move helper function
	void move_helper (iobserver&& other);

	//! a series of dependent subjects that will die when this dies
	std::unordered_set<subject*> ondeath_deps_;

	bool recordable_;

	friend class subject;
};

}

#endif /* TENNCOR_IOBSERVER_HPP */
