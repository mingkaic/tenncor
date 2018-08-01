#include "soil/external.hpp"

DataBucket evaluate (Nodeptr& exit_node)
{
	Session sess;
	return DataBucket(
		exit_node->calculate(sess),
		exit_node->type(),
		exit_node->shape());
}
