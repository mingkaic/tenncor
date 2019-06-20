
%{
#include <stdio.h>
#include <string.h>

#include "experimental/opt/parse/def.h"

extern FILE* yyin;

%}

%start rules

%parse-param {struct PtrList** stmts}

%union
{
	double 				number;
	char 				label[32];
	struct NumList* 	list;
	struct Arg*			argument;
	struct PtrList* 	arguments;
	struct Subgraph* 	sg;
	struct Statement* 	stmt;
}

%type <number> 		NUMBER
%type <label> 		EL
%type <list> 		num_arr
%type <argument> 	arg edge_def key_val
%type <arguments>	args
%type <sg>			subgraph
%type <stmt> 		symbol_def group_def conversion

%token
SYMBOL NEWLINE ARROW LPAREN RPAREN COMMA ASSIGN LSB RSB
LCB RCB GROUP GROUPDEF SHAPER COORDER COLON EL NUMBER VARIADIC

%% /* beginning of rules section */

rules:		/* empty */
			{
				*stmts = new_ptrlist(STATEMENT);
			}
			|
			rules NEWLINE
			{
				// empty line
			}
			|
			rules symbol_def
			{
				// declare symbol
				ptrlist_pushback(*stmts, $2);
			}
			|
			rules group_def
			{
				// declare group
				ptrlist_pushback(*stmts, $2);
			}
			|
			rules conversion
			{
				// declare conversion
				ptrlist_pushback(*stmts, $2);
			}

symbol_def: SYMBOL EL NEWLINE
			{
				char* str = malloc(sizeof($2));
				memcpy(str, $2, sizeof($2));

				struct Statement* stmt = $$ =
					malloc(sizeof(struct Statement));
				stmt->type_ = SYMBOL_DEF;
				stmt->val_ = str;
			}

group_def: 	GROUPDEF EL EL NEWLINE
			{
				struct Group* group = malloc(sizeof(struct Group));
				strncpy(group->ref_, $2, 32);
				strncpy(group->tag_, $3, 32);

				struct Statement* stmt = $$ =
					malloc(sizeof(struct Statement));
				stmt->type_ = GROUP_DEF;
				stmt->val_ = group;
			}

conversion:	subgraph ARROW subgraph NEWLINE
			{
				struct Conversion* cv =
					malloc(sizeof(struct Conversion));
				cv->source_ = $1;
				cv->dest_ = $3;

				struct Statement* stmt = $$ =
					malloc(sizeof(struct Statement));
				stmt->type_ = CONVERSION;
				stmt->val_ = cv;
			}

subgraph:	NUMBER
			{
				// a scalar
				struct Subgraph* sg = $$ =
					malloc(sizeof(struct Subgraph));
				sg->type_ = SCALAR;
				sg->val_.scalar_ = $1;
			}
			|
			EL
			{
				// a symbol
				struct Subgraph* sg = $$ =
					malloc(sizeof(struct Subgraph));
				sg->type_ = ANY;
				char* str = sg->val_.any_ =
					malloc(sizeof($1));
				memcpy(str, $1, sizeof($1));
			}
			|
			GROUP EL LPAREN args RPAREN
			{
				// a group
				struct Subgraph* sg = $$ =
					malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;

				size_t nbranchbytes = sizeof(struct Branch);
				struct Branch* branch = sg->val_.branch_ =
					malloc(nbranchbytes);
				memset(branch, 0, nbranchbytes);

				strncpy(branch->label_, $2, 32);
				branch->is_group_ = 1;
				branch->args_ = $4;
			}
			|
			GROUP EL LPAREN args COMMA VARIADIC EL RPAREN
			{
				// a group with variadic
				struct Subgraph* sg = $$ =
					malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;

				struct Branch* branch = sg->val_.branch_ =
					malloc(sizeof(struct Branch));

				strncpy(branch->label_, $2, 32);
				strncpy(branch->variadic_, $7, 32);
				branch->is_group_ = 1;
				branch->args_ = $4;
			}
			|
			GROUP EL LPAREN VARIADIC EL RPAREN
			{
				// a group with variadic without arguments
				struct Subgraph* sg = $$ =
					malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;

				struct Branch* branch = sg->val_.branch_ =
					malloc(sizeof(struct Branch));

				strncpy(branch->label_, $2, 32);
				strncpy(branch->variadic_, $5, 32);
				branch->is_group_ = 1;
				branch->args_ = new_ptrlist(ARGUMENT);
			}
			|
			EL LPAREN args RPAREN
			{
				// a functor
				struct Subgraph* sg = $$ =
					malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;

				size_t nbranchbytes = sizeof(struct Branch);
				struct Branch* branch = sg->val_.branch_ =
					malloc(nbranchbytes);
				memset(branch, 0, nbranchbytes);

				strncpy(branch->label_, $1, 32);
				branch->is_group_ = 0;
				branch->args_ = $3;
			}

args:		args COMMA arg
			{
				struct PtrList* list = $$ = $1;
				ptrlist_pushback(list, $3);
			}
			|
			arg
			{
				struct PtrList* list = $$ = new_ptrlist(ARGUMENT);
				ptrlist_pushback(list, $1);
			}

arg:		subgraph
			{
				// plain old argument
				size_t nargbytes = sizeof(struct Arg);
				struct Arg* arg = $$ = malloc(nargbytes);
				memset(arg, 0, nargbytes);
				arg->subgraph_ = $1;
			}
			|
			subgraph ASSIGN LCB edge_def RCB
			{
				// argument with edge definition
				struct Arg* arg = $$ = $4;
				arg->subgraph_ = $1;
			}

edge_def:	key_val
			{
				$$ = $1;
			}
			|
			edge_def COMMA key_val
			{
				struct Arg* arg = $$ = $1;
				struct Arg* other = $3;
				if (NULL != other->shaper_)
				{
					if (NULL != arg->shaper_)
					{
						yyerror(stmts, "cannot make multiple definitions "
							"of shaper in edge def");
					}
					arg->shaper_ = other->shaper_;
				}
				if (NULL != other->coorder_)
				{
					if (NULL != arg->coorder_)
					{
						yyerror(stmts, "cannot make multiple definitions "
							"of coorder in edge def");
					}
					arg->coorder_ = other->coorder_;
				}
				arg_recursive_free(other);
			}

key_val:	SHAPER COLON LSB num_arr RSB
			{
				size_t nargbytes = sizeof(struct Arg);
				struct Arg* arg = $$ = malloc(nargbytes);
				memset(arg, 0, nargbytes);
				arg->shaper_ = $4;
			}
			|
			COORDER COLON LSB num_arr RSB
			{
				size_t nargbytes = sizeof(struct Arg);
				struct Arg* arg = $$ = malloc(nargbytes);
				memset(arg, 0, nargbytes);
				arg->coorder_ = $4;
			}

num_arr:	num_arr COMMA NUMBER
			{
				struct NumList* list = $$ = $1;
				numlist_pushback(list, $3);
			}
			|
			NUMBER
			{
				struct NumList* list = $$ = new_numlist();
				numlist_pushback(list, $1);
			}

%%

int parse_rule (struct PtrList** stmts, const char* filename)
{
	int exit_status = 1;
	FILE* file = fopen(filename, "r");
	if (file)
	{
		yyin = file;
		exit_status = yyparse(stmts);
		fclose(file);
	}
	return exit_status;
}

int yyerror (struct PtrList** stmts, char const* msg)
{
	fprintf(stderr, "%s\n", msg);
	return 1;
}