all: 
	cd src; make 
	cd examples; make

clean:
	find . -name "_*" -exec rm {} \;

clean-all: clean
	cd src; make clean
	cd examples; make clean

dist:
	cd .. ; DD=`date +%y-%m-%d`; tar --exclude .svn -zcvf swan-$$DD.tgz swan


