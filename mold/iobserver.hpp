/*!
 *
 *  iobserver.hpp
 *  mold
 *
 *  Purpose:
 *  iobserver audience of inode
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/inode.hpp"

#pragma once
#ifndef MOLD_IOBSERVER_HPP
#define MOLD_IOBSERVER_HPP

namespace mold
{

class iObserver
{
public:
	iObserver (std::vector<iNode*> args);

	virtual ~iObserver (void);

	iObserver (const iObserver& other);

	iObserver (iObserver&& other);

	iObserver& operator = (const iObserver& other);

	iObserver& operator = (iObserver&& other);

	virtual void initialize (void) = 0;

	virtual void update (void) = 0;

	void replace (iNode* target, iNode* repl);

protected:
	std::vector<iNode*> args_;

private:
	void copy_helper (const iObserver& other);

	void move_helper (iObserver&& other);
};

}

#endif /* MOLD_IOBSERVER_HPP */
