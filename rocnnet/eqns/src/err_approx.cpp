#include "llo/api.hpp"

#include "rocnnet/eqns/err_approx.hpp"

DeltasT sgd (const llo::DataNode& root, std::vector<llo::DataNode> leaves,
	double learning_rate)
{
	DeltasT errs;
	for (llo::DataNode& leaf : leaves)
	{
		std::shared_ptr<llo::iSource> leafsrc = leaf.source();
		if (nullptr == leafsrc)
		{
			ade::warnf("attempting to approximate error of root with respect "
				"to context-less tensor %s... skipping",
				leaf.tensor_->to_string().c_str());
			continue;
		}

		// given root = f, err(x) ~ x - η * df(x), where η is the learning rate
		llo::DataNode gres = llo::reduce_sum(root.derive(leaf), 2);
		errs.emplace(leafsrc.get(), llo::sub(leaf,
			llo::mul(gres, llo::Source<double>::get_scalar(learning_rate))));
	}
	return errs;
}

DeltasT rms_momentum (const llo::DataNode& root,
	std::vector<llo::DataNode> leaves, double learning_rate,
	double discount_factor, double epsilon)
{
	DeltasT errs;
	for (llo::DataNode& leaf : leaves)
	{
		std::shared_ptr<llo::iSource> leafsrc = leaf.source();
		if (nullptr == leafsrc)
		{
			ade::warnf("attempting to approximate error of root with respect "
				"to context-less tensor %s... skipping",
				leaf.tensor_->to_string().c_str());
			continue;
		}
		// given root = f, err(x) ~ x - (η * df(x)) / (sqrt(ε + momentum)),
		// where η is the learning rate, and ε is epsilon
		auto gres = llo::reduce_sum(root.derive(leaf), 2);

		// upkeep additional hidden variable momentum: starting with value 1
		// given root = f, err(momentum) ~ χ * momentum + (1 - χ) * df(x) ^ 2,
		// where χ is discount_factor
		ade::Shape shape = leaf.tensor_->shape();
		std::vector<double> wun(shape.n_elems(), 1);
		llo::DataNode momentum = llo::Source<double>::get(shape, wun);
		auto discount_node = llo::Source<double>::get_scalar(discount_factor);
		auto datcount_node = llo::Source<double>::get_scalar(1.0 - discount_factor);
		errs.insert({momentum.source().get(),
			llo::add(llo::mul(discount_node, momentum),
			llo::prod({datcount_node, gres, gres}))});

		errs.emplace(leafsrc.get(), llo::sub(leaf,
			llo::div(llo::mul(gres, llo::Source<double>::get_scalar(learning_rate)),
			llo::add(llo::sqrt(momentum), llo::Source<double>::get_scalar(epsilon)))));
	}
	return errs;
}
