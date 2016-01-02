EXAMPLES = vecadd vecadd_global interp shmem thread chkpt info copy latency
all:
	for T in $(EXAMPLES); do cd $$T; make; cd ..; done

clean:
	for T in $(EXAMPLES); do cd $$T; make clean; cd ..; done

