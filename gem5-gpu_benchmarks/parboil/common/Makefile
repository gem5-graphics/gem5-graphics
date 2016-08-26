# (c) 2007 The Board of Trustees of the University of Illinois.

SRCDIR=$(shell pwd)

OBJS=parboil.o
TARGET=libparboil.a

all : $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) $^ -o $@

$(TARGET) : $(OBJS)
	$(AR) rcs $@ $?

