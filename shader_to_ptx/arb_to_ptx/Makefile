CXX      = g++
CXFLAGS  = 
LEX		= flex
LEXFLAGS	= -B 
YACC		= yacc
YFLAGS		= -t -d -v --report=all
INC_PATHS =   -I$(GPGPUSIM_MESA_ROOT)/Mesa-7.11.2_GPGPU-Sim/src/mesa
INC_PATHS +=  -I$(GPGPUSIM_MESA_ROOT)/Mesa-7.11.2_GPGPU-Sim/src/mapi
INC_PATHS +=  -I$(GPGPUSIM_MESA_ROOT)/Mesa-7.11.2_GPGPU-Sim/src/mesa/main

arb_to_ptx: lex.yy.c y.tab.c y.tab.h
	${CXX} ${CXFLAGS} y.tab.c lex.yy.c ${INC_PATHS} -lfl -o arb_to_ptx

y.tab.c y.tab.h: arb4.y
	${YACC} -d arb4.y

lex.yy.c: arb4.l y.tab.h
	${LEX} arb4.l

clean:
	rm -rf *o arb_to_ptx y.tab.c y.tab.h lex.yy.c
