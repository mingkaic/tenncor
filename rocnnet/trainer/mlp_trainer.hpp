#include "ead/opt/parse.hpp"

#include "rocnnet/modl/mlp.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifndef MODL_MLP_TRAINER_HPP
#define MODL_MLP_TRAINER_HPP

namespace trainer
{

// Normal default context that only stores the number of iterations
struct TrainingContext final : public modl::iTrainingContext
{
	void marshal_layer (cortenn::Layer& out_layer) const override
	{
		cortenn::ItTrainerState* state = out_layer.mutable_it_ctx();
		state->set_iterations(n_iterations_);
	}

	void unmarshal_layer (const cortenn::Layer& in_layer) override
	{
		const cortenn::ItTrainerState& state = in_layer.it_ctx();
		n_iterations_ = state.iterations();
	}

	size_t n_iterations_ = 0;
};

// MLPTrainer does not own anything
struct MLPTrainer
{
	MLPTrainer (modl::MLPptrT brain,
		modl::NonLinearsT nonlinearities,
		ead::Session<PybindT>& sess,
		eqns::ApproxFuncT update, uint8_t batch_size,
		TrainingContext ctx = TrainingContext()) :
		batch_size_(batch_size),
		train_in_(ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({brain->get_ninput(), batch_size}), "train_in")),
		brain_(brain),
		sess_(&sess),
		ctx_(ctx)
	{
		train_out_ = (*brain_)(
			ead::convert_to_node<PybindT>(train_in_), nonlinearities);
		expected_out_ = ead::make_variable_scalar<PybindT>(0.0,
			ade::Shape({brain_->get_noutput(), batch_size}), "expected_out");
		error_ = age::square(
			age::sub(ead::convert_to_node<PybindT>(expected_out_), train_out_));

		pbm::PathedMapT vmap = brain_->list_bases();
		eqns::VariablesT vars;
		for (auto vpair : vmap)
		{
			if (auto var = std::dynamic_pointer_cast<
				ead::Variable<PybindT>>(vpair.first))
			{
				vars.push_back(
					std::make_shared<ead::VariableNode<PybindT>>(var));
			}
		}
		updates_ = update(error_, vars);

		std::vector<ead::NodeptrT<PybindT>*> to_optimize =
		{
			&train_out_,
			&error_,
		};
		for (eqns::AssignsT& assigns : updates_)
		{
			for (eqns::VarAssign& assign : assigns)
			{
				to_optimize.push_back(&assign.source_);
			}
		}

		size_t n_roots = to_optimize.size();
		ead::NodesT<PybindT> roots(n_roots);
		std::transform(to_optimize.begin(), to_optimize.end(), roots.begin(),
			[](ead::NodeptrT<PybindT>* ptr)
			{
				return *ptr;
			});
		{
			auto rules = ead::opt::get_configs<PybindT>();
			ead::opt::optimize(roots, rules);
		}

		for (size_t i = 0; i < n_roots; ++i)
		{
			sess_->track(roots[i]);
			*to_optimize[i] = roots[i];
		}
	}

	void train (std::vector<PybindT>& train_in,
		std::vector<PybindT>& expected_out)
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
		assign_groups(updates_,
			[this](std::unordered_set<ade::iTensor*>& updated)
			{
				this->sess_->update(updated);
			});
		++ctx_.n_iterations_;
	}

	bool save (std::ostream& outs)
	{
		return modl::save(outs,
			error_->get_tensor(), brain_.get(), &ctx_);
	}

	uint8_t batch_size_;
	ead::VarptrT<PybindT> train_in_;
	ead::VarptrT<PybindT> expected_out_;
	modl::MLPptrT brain_;
	ead::NodeptrT<PybindT> train_out_;
	ead::NodeptrT<PybindT> error_;

	eqns::AssignGroupsT updates_;
	ead::Session<PybindT>* sess_;

	TrainingContext ctx_;
};

}

#endif // MODL_MLP_TRAINER_HPP
