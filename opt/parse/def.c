#include <stdio.h>

#include "opt/parse/def.h"

#ifdef PARSE_DEF_H

void subgraph_recursive_free (void* ptr)
{
	if (NULL == ptr)
	{
		return;
	}
	struct Subgraph* sg = (struct Subgraph*) ptr;
	switch (sg->type_)
	{
		case SCALAR:
			break;
		case ANY:
			free(sg->val_.any_);
			break;
		case BRANCH:
		{
			struct Branch* branch = sg->val_.branch_;
			ptrlist_free(branch->args_, &arg_recursive_free);
			free(branch);
		}
			break;
		default:
			fprintf(stderr, "freeing unknown subgraph %d\n", sg->type_);
	}
	free(sg);
}

void arg_recursive_free (void* ptr)
{
	if (NULL == ptr)
	{
		return;
	}
	struct Arg* arg = (struct Arg*) ptr;
	subgraph_recursive_free(arg->subgraph_);
	numlist_free(arg->shaper_);
	numlist_free(arg->coorder_);
	free(arg);
}

void conversion_recursive_free (void* ptr)
{
	if (NULL == ptr)
	{
		return;
	}
	struct Conversion* conv = (struct Conversion*) ptr;
	subgraph_recursive_free(conv->source_);
	subgraph_recursive_free(conv->dest_);
	free(conv);
}

void statement_recursive_free (void* ptr)
{
	if (NULL == ptr)
	{
		return;
	}
	struct Statement* stmt = (struct Statement*) ptr;
	switch (stmt->type_)
	{
		case SYMBOL_DEF:
		case PROPERTY_DEF:
			free(stmt->val_);
			break;
		case CONVERSION:
			conversion_recursive_free(stmt->val_);
			break;
		default:
			fprintf(stderr, "freeing unknown statement type %d", stmt->type_);
	}
	free(stmt);
}

void statements_free (struct PtrList* stmts)
{
	ptrlist_free(stmts, statement_recursive_free);
}

#endif
