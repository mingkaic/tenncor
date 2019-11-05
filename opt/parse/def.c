#include <stdio.h>

#include "opt/parse/def.h"

#ifdef PARSE_DEF_H

void subgraph_recursive_free (struct Subgraph* sg)
{
	if (NULL == sg)
	{
		return;
	}
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

void arg_recursive_free (void* arg)
{
	if (NULL == arg)
	{
		return;
	}
	struct Arg* argt = (struct Arg*) arg;
	subgraph_recursive_free(argt->subgraph_);
	numlist_free(argt->shaper_);
	numlist_free(argt->coorder_);
	free(argt);
}

void conversion_recursive_free (void* conv)
{
	if (NULL == conv)
	{
		return;
	}
	struct Conversion* convt = (struct Conversion*) conv;
	subgraph_recursive_free(convt->source_);
	subgraph_recursive_free(convt->dest_);
	free(convt);
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
