%{
#include <string.h>
#include <stdlib.h>

#include "opt/parse/def.h"
#include "opt/parse/rulecfg_y.h"

%}

%option nodefault
%option noyywrap

%%
[\t\r\a\v\b\n ]					; // spaces

\/\/.*							; // comments

symbol							{ return SYMBOL; }

;								{ return STMT_TERM; }

=>								{ return ARROW; }

\(								{ return LPAREN; }

\)								{ return RPAREN; }

,								{ return COMMA; }

=								{ return ASSIGN; }

\[ 								{ return LSB; }

\]								{ return RSB; }

\{ 								{ return LCB; }

\}								{ return RCB; }

property						{ return PROPERTY; }

group							{ return GROUP; }

shaper							{ return SHAPER; }

coorder							{ return COORDER; }

:								{ return COLON; }

\.\.							{ return VARIADIC; }

[a-zA-Z\_][a-zA-Z\_0-9]{0,31}	{
	strncpy(yylval.label, yytext, 32);
	return EL;
}

-?(0|[1-9][0-9]*)(\.\d+)?		{
	yylval.number = atof(yytext);
	return NUMBER;
}

.								{
	fprintf(stderr, "unrecognized token %s", yytext);
}

%%
