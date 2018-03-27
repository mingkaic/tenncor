//
// Created by Mingkai Chen on 2017-04-27.
//

#include "include/graph/variable.hpp"
#include "include/operate/operations.hpp"

#pragma once
#define TENNCOR_GD_UTILS_HPP
#ifndef TENNCOR_GD_UTILS_HPP
#define TENNCOR_GD_UTILS_HPP


namespace nnet
{

using UPDATE_F = std::function<void(bool)>;

using UPDATES_T = std::vector<UPDATE_F>;

using GINTERM_F = std::function<varptr(varptr,variable*)>;

//! gradient descent algorithm abstraction
class gd_updater
{
public:
	gd_updater (double learning_rate);

	virtual ~gd_updater(void);

	gd_updater* clone (void) const;

	gd_updater* move (void);

	virtual UPDATES_T calculate (inode* root, GINTERM_F intermediate_process =
		[](varptr grad, variable*) { return grad; });

	void ignore_subtree (inode* subroot);

	void clear_ignore (void);
	
	void set_learning_rate (double learning_rate);

protected:
	virtual gd_updater* clone_impl (void) const = 0;

	virtual gd_updater* move_impl (void) = 0;

	virtual UPDATE_F process_update (varptr& gres,
		variable* leaf, GINTERM_F intermediate_process) = 0;

	double learning_rate_;
	
private:
	std::unordered_set<variable*> ignored_;
};

//! vanilla gradient descent algorithm
class vgb_updater : public gd_updater
{
public:
	vgb_updater (double learning_rate = 0.5);

	vgb_updater* clone (void);

	vgb_updater* move (void);

protected:
	virtual gd_updater* clone_impl (void) const;

	virtual gd_updater* move_impl (void);

	virtual UPDATE_F process_update (varptr& gres,
		variable* leaf, GINTERM_F intermediate_process);
};

//! momentum based gradient descent
//! overview: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
//! updates using incremental update of velocity on error manifold

// Standard momentum:
// 1. velocity_t = discount * velocity_(t-1) - learning * J'[v]
// 2. delta(var) = velocity_t, update by gd
// lim t->inf (velocity) = -learning * J'[v] / (1-discount)

// Nestrov momentum:
// 1. delta(var) = velocity_t-1, update by gd
// 2. velocity_t = discount * velocity_(t-1) - learning * J'[v]
class momentum_updater : public gd_updater
{
public:
	momentum_updater (double learning_rate = 0.5) : gd_updater(learning_rate) {}
	
	momentum_updater* clone (void);

	momentum_updater* move (void);

protected:
	virtual gd_updater* clone_impl (void) const;

	virtual gd_updater* move_impl (void);

	virtual UPDATE_F process_update (varptr& gres,
		variable* leaf, GINTERM_F intermediate_process);
};

// Separate adaptive learning rates
// introduce leaf local_gain linked to weight/bias variables
// delta(var) = -epsilon * local_gain[v] * J'[v]
// if J'[v]_t * J'[v]_(t-1)> 0:
// then local_gain[v] += 0.05
// else local_gain[v] *= 0.95
class adadelta_updater : public gd_updater
{
public:
	adadelta_updater (double learning_rate = 0.5) : gd_updater(learning_rate) {}
	
	adadelta_updater* clone (void);

	adadelta_updater* move (void);

	double rho_ = 0.95;

	double epsilon_ = std::numeric_limits<double>::epsilon();

protected:
	virtual gd_updater* clone_impl (void) const;

	virtual gd_updater* move_impl (void);

	virtual UPDATE_F process_update (varptr& gres,
		variable* leaf, GINTERM_F intermediate_process);
};

// adaptive gradient
class adagradupdater : public gd_updater
{
public:
	adagradupdater (double learning_rate = 0.5) : gd_updater(learning_rate) {}
	
	adagradupdater* clone (void);

	adagradupdater* move (void);

	double init_accum_ = 0.1;

protected:
	virtual gd_updater* clone_impl (void) const;

	virtual gd_updater* move_impl (void);

	virtual UPDATE_F process_update (varptr& gres,
		variable* leaf, GINTERM_F intermediate_process);
};


//! RMS prop
// rms_delta = J'(v)_t
// rms_t = (1 - discount) * rms_t-1 + discount * rms_delta^2
// delta(var) = v_t = learning * rms_delta / rms_t
// initial momentum is 1 (todo: decide whether or not parameterizing init momentum matters)
class rmspropupdater : public gd_updater
{
public:
	rmspropupdater (double learning_rate = 0.5, double discount_factor = 0.99);

	virtual ~rmspropupdater (void);

	rmspropupdater* clone (void);

	rmspropupdater* move (void);
	
	virtual rmspropupdater& operator = (const rmspropupdater& other);
	
	virtual rmspropupdater& operator = (rmspropupdater&& other);
	
	void set_discount_factor (double discount_factor);

protected:
	// never copy or move momentums_
	rmspropupdater (const rmspropupdater& other);
	
	rmspropupdater (rmspropupdater&& other);

	virtual gd_updater* clone_impl (void) const;

	virtual gd_updater* move_impl (void);

	virtual UPDATE_F process_update (varptr& gres,
		variable* leaf, GINTERM_F intermediate_process);
		
private:
	double discount_factor_;

	std::vector<variable*> momentums_;

	const double epsilon_ = std::numeric_limits<double>::epsilon();
};

}


#endif /* TENNCOR_GD_UTILS_HPP */

#undef TENNCOR_GD_UTILS_HPP