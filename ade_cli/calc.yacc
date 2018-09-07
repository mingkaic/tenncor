%{
#include<stdio.h>

#include "ade_cli/ast/calc_ast.h"

char buffer[64];
int base;

extern int yylex();
extern int yyerror();
extern FILE *yyin;

%}

%start list

%union
{
	int int_type;
	char str_type[32];
	struct ShapeHolder* vec_type;
	struct ASTNode* ref_type;
}

%type <int_type> number UNARY BINARY SHAPEOP DIGIT
%type <str_type> VAR
%type <vec_type> shape
%type <ref_type> expr

%token DIGIT VAR UNARY BINARY SHAPEOP GRAD SHOW_EQ EXIT

%left '+' '-'
%left '*' '/'
%left UMINUS /*supplies precedence for unary minus */

%% /* beginning of rules section */

list: 	/* empty */
		|
		list EXIT
		{
			YYACCEPT;
		}
		|
		list stat '\n'
		|
		list error '\n'
		{
			yyerrok;
		}

stat:	/* empty */
		{
			printf("\n");
		}
		|
		expr
		{
			to_string(buffer, $1);
			printf("%s\n", buffer);
			free_ast($1);
		}
		|
		VAR '=' expr
		{
			saveAST($1, $3);
			free_ast($3);
		}
		|
		SHOW_EQ '(' expr ')'
		{
			show_eq($3);
			free_ast($3);
		}
		;

expr:   GRAD '(' expr ',' VAR ')'
		{
			struct ASTNode* node = loadAST($5);
			$$ = grad($3, node);
			free_ast($3);
			free_ast(node);

		}
		|
		SHAPEOP '(' expr ',' shape ')'
		{
			$$ = shapeop($3, $5, $1);
			free_ast($3);
			free_shape($5);
		}
		|
		BINARY '(' expr ',' expr ')'
		{
			$$ = binary($3, $5, $1);
			free_ast($3);
			free_ast($5);
		}
		|
		UNARY '(' expr ')'
		{
			$$ = unary($3, $1);
			free_ast($3);
		}
		|
		'(' expr ')'
		{
			$$ = $2;
		}
		|
		expr '*' expr
		{
			$$ = binary($1, $3, MUL);
			free_ast($1);
			free_ast($3);
		}
		|
		expr '/' expr
		{
			$$ = binary($1, $3, DIV);
			free_ast($1);
			free_ast($3);
		}
		|
		expr '+' expr
		{
			$$ = binary($1, $3, ADD);
			free_ast($1);
			free_ast($3);
		}
		|
		expr '-' expr
		{
			$$ = binary($1, $3, SUB);
			free_ast($1);
			free_ast($3);
		}
		|
		'-' expr %prec UMINUS
		{
			$$ = unary($2, NEG);
			free_ast($2);
		}
		|
		VAR
		{
			$$ = loadAST($1);
		}
		|
		'[' shape ']'
		{
			$$ = toNode($2);
			free_shape($2);
		}
		|
		'[' ']'
		{
			$$ = emptyNode();
		}
		;

shape:	shape ',' number
		{
			append($1, $3);
			$$ = $1;
		}
		|
		number
		{
			$$ = make($1);
		}
		;

number: DIGIT
		{
		   $$ = $1;
		   base = ($1==0) ? 8 : 10;
		}
		|
		number DIGIT
		{
		   $$ = base * $1 + $2;
		}
		;

%%
int main (int argc, char** argv)
{
	int exit_status = 0;
	if (argc > 1)
	{
		int i;
		for (i = 1; i < argc; ++i)
		{
			FILE* file = fopen(argv[i], "r");
			if (file)
			{
				yyin = file;
				exit_status &= yyparse();
				fclose(file);
			}
			else
			{
				// warn
			}
		}
	}
	else
	{
		exit_status = yyparse();
	}
	return exit_status;
}

int yyerror(s)
char *s;
{
	// todo: improve logger
	fprintf(stderr, "%s\n",s);
}

int yywrap()
{
	return(1);
}
