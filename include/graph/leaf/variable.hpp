/*!
 *
 *  variable.hpp
 *  cnnet
 *
 *  Purpose:
 *  define the graph variable implementation
 *
 *  Created by Mingkai Chen on 2017-02-27.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_VARIABLE_HPP
#define TENNCOR_VARIABLE_HPP

namespace nnet
{

using variable_updater = std::function<void(bool)>;

struct open_source : public idata_source
{
	open_source (std::shared_ptr<void> defsrc) : source_(defsrc) {}

	virtual idata_source* clone (void)
	{
		return new open_source(*this);
	}

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape)
	{
		assert(nullptr != source_);
		return source_->get_data(type, shape);
	}

	shared_ptr<idata_source> source_;
};

struct assign_io : virtual idata_source, virtual idata_dest
{
	virtual void set_data (std::shared_ptr<void> data, 
		TENS_TYPE type, tensorshape shape, size_t i)
	{
		assert(type_ == BAD_T || type_ == type);
		size_t nargs = args_.size();
		if (i < nargs)
		{
			args_.insert(args_.end(), args_.size() - i, nullptr);
			args_shape_.insert(args_shape_.end(), args_.size() - i, tensorshape);
		}
		args_[i] = data;
		args_shape_[i] = shape;
	}

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape)
	{
		type = type_;
		// todo: check for shapes
		if (!opname_.empty())
		{
			return args_[0];
		}
		std::vector<VARR> args;
		for (size_t i = 0; i < args_.size(); i++)
		{
			args.push_back(VARR{args_[i].get(), args_shape_[i]});
		}
		operate(opname_, type_, dest_, args);
	}

	void clear (void)
	{
		opname_.clear();
		dest_ = nullptr;
		args_.clear();
		args_shape_.clear();
		type_ = BAD_T;
	}

	std::string opname_;
	std::shared_ptr<void> dest_;
	std::vector<std::shared_ptr<void> > args_;
	std::vector<tensorshape> args_shape_;
	TENS_TYPE type_;
};

class variable final : public ileaf
{
public:
	// >>>> CONSTRUCTORS <<<<
	//! construct to init zero and one
	variable (const tensorshape& shape, 
		idata_source* source, std::string name);

	//! copy construct to init zero and one
	variable (const variable& other);

	//! move construct to init zero and one
	variable (variable&& other);

	virtual ~variable (void);

	//! clone function
	variable* clone (void) const;

	//! move function
	variable* move (void);

	//! declare copy assignment to copy over initializer
	virtual variable& operator = (const variable& other);

	//! declare move assignment to move over initializer
	virtual variable& operator = (variable&& other);



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node
	virtual varptr derive (inode* wrt);

	// >>>>>> VARIABLE SPECIAL <<<<<<

	//! initialize data, return true if success
	void initialize (void);

	//! initialize data using shape, 
	//! return true if success
	void initialize (tensorshape shape);

	//! assign contents of input to this, return true if successful
	bool assign (inode* input, bool notify = true);

	//! return update data function (add input node data to this)
	variable_updater assign_add (inode* input);

	//! return update data function (subtract input node data to this)
	variable_updater assign_sub (inode* input);

protected:
	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	void variable::copy_helper (const variable& other);

	void variable::move_helper (variable&& other);

private:
	std::shared_ptr<assign_io> asgn_ = std::make_shared<assign_io>();

	std::shared_ptr<open_source> dsrc_;

	//! raw data
	std::unique_ptr<tensor> data_ = nullptr;
};

}

#endif /* TENNCOR_VARIABLE_HPP */
