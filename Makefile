NAME		=       nbody
CC		=	gcc
CXX		=	icpc
LD		=	gcc
DEBUG		+=
CFLAGS		+=	-O3 $(DEBUG) -fopenmp
CXXFLAGS	+=	$(CFLAGS)
LDFLAGS		+=	-lm

EXE		=	$(NAME)
CFILES		=	$(wildcard *.c)
CXXFILES	=	$(wildcard *.cpp)
OBJECTS		=	$(CFILES:.c=.o) $(CXXFILES:.cpp=.o)

all : $(EXE)
	@echo "Building..."

$(EXE) : $(OBJECTS)
	$(LD) $^ $(CFLAGS) $(LDFLAGS) -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean :	
	rm -fr $(EXE) $(OBJECTS) 
