#include "rocnnet/modl/mlp.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifndef MODL_mlp_trainer_HPP
#define MODL_mlp_trainer_HPP

// MLPTrainer does not own anything
struct MLPTrainer
{
	MLPTrainer (modl::MLPptrT brain, ead::Session<double>& sess,
		eqns::ApproxFuncT update, uint8_t batch_size) :
		batch_size_(batch_size),
		train_in_(ead::make_variable_scalar<double>(0.0,
			ade::Shape({brain->get_ninput(), batch_size}), "train_in")),
		brain_(brain),
		sess_(&sess)
	{
		train_out_ = (*brain_)(ead::convert_to_node<double>(train_in_));
		expected_out_ = ead::make_variable_scalar<double>(0.0,
			ade::Shape({brain_->get_noutput(), batch_size}), "expected_out");
		error_ = age::square(age::sub(ead::convert_to_node<double>(expected_out_), train_out_));

		pbm::PathedMapT vmap = brain_->list_bases();
		eqns::VariablesT vars;
		for (auto vpair : vmap)
		{
			if (auto var = std::dynamic_pointer_cast<
				ead::Variable<double>>(vpair.first))
			{
				vars.push_back(
					std::make_shared<ead::VariableNode<double>>(var));
			}
		}
		updates_ = update(error_, vars);
		for (ead::NodeptrT<double>& up : updates_.upkeep_)
		{
			sess_->track(up);
		}
		sess_->track(train_out_);
		sess_->track(error_);
	}

	void train (std::vector<double>& train_in,
		std::vector<double>& expected_out)
	{
		size_t insize = brain_->get_ninput();
		size_t outsize = brain_->get_noutput();
		if (train_in.size() != insize * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match "
				"input size (%d) * batchsize (%d)", train_in.size(),
				insize, batch_size_);
		}
		if (expected_out.size() != outsize * batch_size_)
		{
			logs::fatalf("expected output size (%d) does not match "
				"output size (%d) * batchsize (%d)", expected_out.size(),
				outsize, batch_size_);
		}
		train_in_->assign(train_in.data(), train_in_->shape());
		expected_out_->assign(expected_out.data(), expected_out_->shape());

		sess_->update({
			train_in_->get_tensor().get(),
			expected_out_->get_tensor().get(),
		});
		updates_.assign(*sess_);
	}

	uint8_t batch_size_;
	ead::VarptrT<double> train_in_;
	ead::VarptrT<double> expected_out_;
	modl::MLPptrT brain_;
	ead::NodeptrT<double> train_out_;
	ead::NodeptrT<double> error_;

	eqns::Deltas updates_;
	ead::Session<double>* sess_;
};

#endif // MODL_mlp_trainer_HPP
