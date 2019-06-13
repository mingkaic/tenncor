
%{
#include <stdio.h>
#include <string.h>

#include "experimental/opt/parse/def.h"

extern FILE* yyin;

%}

%start rules

%parse-param {struct StmtList* stmts}

%union
{
	double dec_type;
	char str_type[32];
	struct NumList* arr_type;
	struct Arg* arg_type;
	struct ArgList* arg_arr_type;
	struct Subgraph* sg_type;
	struct StmtList* stmt_type;
}

%type <dec_type> NUMBER
%type <str_type> EL
%type <arr_type> num_arr
%type <arg_type> key_val edge_def arg
%type <arg_arr_type> args
%type <sg_type> subgraph
%type <stmt_type> conversion symbol_def rules

%token
SYMBOL NEWLINE ARROW LPAREN RPAREN COMMA ASSIGN LSB RSB
LCB RCB GTAG SHAPER COORDER COLON EL NUMBER VARIADIC

%% /* beginning of rules section */

rules:		/* empty */
			{
				stmts = yylval.stmt_type = NULL;
			}
			|
			symbol_def rules
			{
				// declare symbol
				stmts = yylval.stmt_type = $1;
				stmts->next_ = $2;
			}
			|
			conversion rules
			{
				// declare conversion
				stmts = yylval.stmt_type = $1;
				stmts->next_ = $2;
			}

symbol_def: SYMBOL EL NEWLINE
			{
				char* str = malloc(sizeof($2));
				memcpy(str, $2, sizeof($2));
				struct StmtList* list = yylval.stmt_type = malloc(sizeof(struct StmtList));
				list->next_ = NULL;
				list->type_ = SYMBOL_DEF;
				list->val_ = str;
			}

conversion:	subgraph ARROW subgraph NEWLINE
			{
				struct Conversion* cv = malloc(sizeof(struct Conversion));
				cv->source_ = $1;
				cv->dest_ = $3;
				struct StmtList* list = yylval.stmt_type = malloc(sizeof(struct StmtList));
				list->next_ = NULL;
				list->type_ = CONVERSION;
				list->val_ = cv;
			}

subgraph:	NUMBER
			{
				// a scalar
				struct Subgraph* sg = yylval.sg_type = malloc(sizeof(struct Subgraph));
				sg->type_ = SCALAR;
				sg->val_.scalar_ = $1;
			}
			|
			EL
			{
				// a symbol
				struct Subgraph* sg = yylval.sg_type = malloc(sizeof(struct Subgraph));
				sg->type_ = ANY;
				char* str = sg->val_.any_ = malloc(sizeof($1));
				memcpy(str, $1, sizeof($1));
			}
			|
			GTAG COLON EL LPAREN args RPAREN
			{
				// a group
				struct Subgraph* sg = yylval.sg_type = malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;
				struct Branch* branch = sg->val_.branch_ = malloc(sizeof(struct Branch));
				branch->is_group_ = 1;
				branch->args_ = $5;
				memcpy(branch->label_, $3, sizeof($3));
				memset(branch->variadic_, '\0', sizeof(branch->variadic_));
			}
			|
			GTAG COLON EL LPAREN args COMMA VARIADIC EL RPAREN
			{
				// a group with variadic
				struct Subgraph* sg = yylval.sg_type = malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;
				struct Branch* branch = sg->val_.branch_ = malloc(sizeof(struct Branch));
				branch->is_group_ = 1;
				branch->args_ = $5;
				memcpy(branch->label_, $3, sizeof($3));
				memcpy(branch->variadic_, $8, sizeof($8));
			}
			|
			EL LPAREN args RPAREN
			{
				// a functor
				struct Subgraph* sg = yylval.sg_type = malloc(sizeof(struct Subgraph));
				sg->type_ = BRANCH;
				struct Branch* branch = sg->val_.branch_ = malloc(sizeof(struct Branch));
				branch->is_group_ = 0;
				branch->args_ = $3;
				strncpy(branch->label_, $1, 32);
			}

args:		args COMMA arg
			{
				struct ArgList* arr = yylval.arg_arr_type = $1;
				arr = arr->next_ = malloc(sizeof(struct ArgList));
				arr->next_ = NULL;
				arr->val_ = $3;
			}
			|
			arg
			{
				struct ArgList* arr = yylval.arg_arr_type = malloc(sizeof(struct ArgList));
				arr->next_ = NULL;
				arr->val_ = $1;
			}

arg:		subgraph
			{
				// plain old argument
				struct Arg* arg = yylval.arg_type = malloc(sizeof(struct Arg));
				memset(arg, NULL, sizeof(struct Arg));
				arg->subgraph_ = $1;
			}
			|
			subgraph ASSIGN LCB edge_def RCB
			{
				// argument with edge definition
				struct Arg* arg = yylval.arg_type = $4;
				arg->subgraph_ = $1;
			}

edge_def:	key_val
			{
				yylval.arg_type = $1;
			}
			|
			edge_def COMMA key_val
			{
				struct Arg* arg = yylval.arg_type = $1;
				struct Arg* other = $3;
				if (NULL != other->shaper_)
				{
					arg->shaper_ = other->shaper_;
				}
				if (NULL != other->coorder_)
				{
					arg->coorder_ = other->coorder_;
				}
				arg_recursive_free(other);
			}

key_val:	SHAPER COLON num_arr
			{
				struct Arg* arg = yylval.arg_type = malloc(sizeof(struct Arg));
				memset(arg, NULL, sizeof(struct Arg));
				arg->shaper_ = $3;
			}
			COORDER COLON num_arr
			{
				struct Arg* arg = yylval.arg_type = malloc(sizeof(struct Arg));
				memset(arg, NULL, sizeof(struct Arg));
				arg->coorder_ = $3;
			}

num_arr:	num_arr COMMA NUMBER
			{
				struct NumList* arr = yylval.arr_type = $1;
				arr = arr->next_ = malloc(sizeof(struct NumList));
				arr->next_ = NULL;
				arr->val_ = $3;
			}
			|
			NUMBER
			{
				struct NumList* arr = yylval.arr_type = malloc(sizeof(struct NumList));
				arr->next_ = NULL;
				arr->val_ = $1;
			}

%%

int parse_rule (struct StmtList** stmts, const char* filename)
{
	int exit_status = 1;
	FILE* file = fopen(filename, "r");
	if (file)
	{
		yyin = file;
		exit_status = yyparse(*stmts);
		fclose(file);
	}
	return exit_status;
}
