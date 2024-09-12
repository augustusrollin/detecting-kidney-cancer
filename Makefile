OS := $(shell uname -s)

ifeq ($(OS), Darwin)
    include Makefile.mac
else
    include Makefile.win
endif
