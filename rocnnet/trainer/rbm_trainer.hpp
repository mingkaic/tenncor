#include "rocnnet/modl/rbm.hpp"

#include "rocnnet/eqns/err_approx.hpp"

#ifndef MODL_RBM_TRAINER_HPP
#define MODL_RBM_TRAINER_HPP

namespace trainer
{

eqns::AssignGroupsT bbernoulli_approx (const eqns::VarErrsT& leaves,
	PybindT learning_rate, PybindT discount_factor,
	std::string root_label = "")
{
	// assign momentums before leaves
	eqns::AssignsT momentum_assigns;
	eqns::AssignsT leaf_assigns;
	for (size_t i = 0, nleaves = leaves.size(); i < nleaves; ++i)
	{
		auto leaf_node = ead::convert_to_node(leaves[i].first);
		auto err = leaves[i].second;

		auto momentum = ead::make_variable_scalar<PybindT>(0,
			err->shape(), leaves[i].first->get_label() + "_momentum");
		auto momentum_next = tenncor::add(
			tenncor::mul(
				ead::make_constant_scalar(discount_factor, momentum->shape()),
				ead::convert_to_node(momentum)),
			tenncor::mul(
				ead::make_constant_scalar(learning_rate *
					(1 - discount_factor) / err->shape().at(0), err->shape()),
				err));
		auto leaf_next = tenncor::add(leaf_node, momentum_next);

		momentum_assigns.push_back(eqns::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_momentum_%s",
				root_label.c_str(), leaves[i].first->get_label().c_str()),
			momentum, momentum_next});
		leaf_assigns.push_back(eqns::VarAssign{
			fmts::sprintf("bbernoulli_momentum::%s_grad_%s",
				root_label.c_str(), leaves[i].first->get_label().c_str()),
			leaves[i].first, leaf_next});
	}
	return {momentum_assigns, leaf_assigns};
}

using ErrorF = std::function<ead::NodeptrT<PybindT>(ead::NodeptrT<PybindT>,ead::NodeptrT<PybindT>)>;

struct BernoulliRBMTrainer final
{
	BernoulliRBMTrainer (modl::RBM& model,
		ead::iSession& sess,
		ade::DimT batch_size,
		PybindT learning_rate,
		PybindT discount_factor,
		ErrorF err_func = ErrorF()) :
		model_(model), sess_(&sess), batch_size_(batch_size)
	{
		visible_ = ead::make_variable_scalar<PybindT>(0,
			ade::Shape({(ade::DimT) model.get_ninput(), batch_size}));
		expect_hidden_ = ead::make_variable_scalar<PybindT>(0,
			ade::Shape({(ade::DimT) model.get_noutput(), batch_size}));

		hidden_sample_ = model.connect(visible_);
		visible_sample_ = model.backward_connect(
			tenncor::random::rand_binom_one(hidden_sample_));

		auto hidden_reconp = model.connect(visible_sample_);

		auto grad_w = tenncor::sub(
			tenncor::matmul(tenncor::transpose(
				ead::convert_to_node(visible_)), hidden_sample_),
			tenncor::matmul(tenncor::transpose(visible_sample_), hidden_reconp));
		auto grad_hb = tenncor::reduce_mean_1d(
			tenncor::sub(hidden_sample_, hidden_reconp), 1);
		auto grad_vb = tenncor::reduce_mean_1d(
			tenncor::sub(ead::convert_to_node(visible_), visible_sample_), 1);

		auto contents = model.get_contents();
		std::vector<ead::VarptrT<PybindT>> vars;
		vars.reserve(contents.size());
		std::transform(contents.begin(), contents.end(),
			std::back_inserter(vars),
			[](ade::TensptrT tens)
			{
				return std::make_shared<ead::VariableNode<PybindT>>(
					std::static_pointer_cast<ead::Variable<PybindT>>(tens));
			});
		eqns::VarErrsT varerrs = {
			{vars[0], grad_w},
			{vars[1], grad_hb},
			{vars[2], grad_vb},
		};

		updates_ = bbernoulli_approx(varerrs, learning_rate, discount_factor);
		visible_from_hidden_ = model.backward_connect(expect_hidden_);

		if (err_func)
		{
			error_ = err_func(ead::convert_to_node(visible_), visible_sample_);
		}
	}

	PybindT train (std::vector<PybindT>& train_in)
	{
		size_t insize = model_.get_ninput();
		size_t outsize = model_.get_noutput();
		if (train_in.size() != insize * batch_size_)
		{
			logs::fatalf("training vector size (%d) does not match "
				"input size (%d) * batchsize (%d)", train_in.size(),
				insize, batch_size_);
		}
		visible_->assign(train_in.data(), visible_->shape());

		sess_->update({
			visible_->get_tensor().get(),
		});
		assign_groups(updates_,
			[this](std::unordered_set<ade::iTensor*>& updated)
			{
				this->sess_->update(updated);
			});

		if (nullptr == error_)
			return -1;
		return error_->data()[0];
	}

	modl::RBM& model_;

	ead::VarptrT<PybindT> visible_ = nullptr;

	ead::VarptrT<PybindT> expect_hidden_ = nullptr;

	ead::NodeptrT<PybindT> hidden_sample_ = nullptr;

	ead::NodeptrT<PybindT> visible_sample_ = nullptr;

	ead::NodeptrT<PybindT> visible_from_hidden_ = nullptr;

	ead::NodeptrT<PybindT> error_ = nullptr;

	// === updates && optimizer ===
	eqns::AssignGroupsT updates_;

	ead::iSession* sess_;

	size_t batch_size_;
};

}

#endif // MODL_RBM_TRAINER_HPP
