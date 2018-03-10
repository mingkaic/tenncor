//
// Created by Mingkai Chen on 2017-03-17.
//

#ifndef TENNCOR_MOCK_CONNECTOR_H
#define TENNCOR_MOCK_CONNECTOR_H

#include <algorithm>

#include "tests/unit/include/utils/util_test.hpp"
#include "tests/unit/include/utils/mockerino.h"

#include "include/graph/connector/iconnector.hpp"

using namespace nnet;


class mock_connector : public iconnector, public mocker
{
public:
	mock_connector (std::vector<inode*> dependencies, std::string label) :
		iconnector(dependencies, label) {}

	mock_connector (const mock_connector& other) :
		iconnector(other) {}

	mock_connector (mock_connector&& other) :
		iconnector(std::move(other)) {}

	mock_connector& operator = (const mock_connector& other)
	{
		iconnector::operator = (other);
		return *this;
	}

	mock_connector& operator = (mock_connector&& other)
	{
		iconnector::operator = (std::move(other));
		return *this;
	}

	void* get_gid (void) { return this->g_man_; }

	virtual void temporary_eval (const iconnector*,inode*&) const {}
	virtual tensorshape get_shape (void) const { return tensorshape(); }
	virtual const tensor* get_eval (void) const { return nullptr; }
	virtual varptr derive (inode*) { return nullptr; }
	virtual bool read_proto (const tenncor::tensor_proto&) { return false; }

	virtual void update (void)
	{
		label_incr("update1");
	}

	virtual void death_on_broken (void)
	{
		label_incr("death_on_broken");
	}
	virtual std::unordered_set<inode*> get_leaves (void) const
	{
		label_incr("get_leaves1");
		return std::unordered_set<inode*>{};
	}

protected:
	virtual inode* clone_impl (void) const
	{
		return new mock_connector(*this);
	}
	virtual inode* move_impl (void)
	{
		return new mock_connector(std::move(*this));
	}
	virtual inode* get_gradient (variable*) { return nullptr; }
};


#endif //TENNCOR_MOCK_CONNECTOR_H

