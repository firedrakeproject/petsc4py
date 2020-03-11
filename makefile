.PHONY: default
default: build

package = petsc4py
MODULE  = PETSc

PYTHON  = python$(py)
MPIEXEC = mpiexec

# ----

.PHONY: config build test
config:
	${PYTHON} setup.py config ${CONFIGOPT}
build:
	${PYTHON} setup.py build ${BUILDOPT}
test:
	${VALGRIND} ${PYTHON} ${PWD}/test/runtests.py
test-%:
	${MPIEXEC} -n $* ${VALGRIND} ${PYTHON} ${PWD}/test/runtests.py

.PHONY: srcbuild srcclean
srcbuild:
	${PYTHON} setup.py build_src ${SRCOPT}
srcclean:
	-${RM} src/${package}.${MODULE}.c
	-${RM} src/include/${package}/${package}.${MODULE}.h
	-${RM} src/include/${package}/${package}.${MODULE}_api.h
	-${RM} src/lib${package}/lib${package}.[ch]

.PHONY: clean distclean fullclean
clean:
	${PYTHON} setup.py clean --all
distclean: clean
	-${RM} -r build  _configtest.* *.py[co]
	-${RM} -r MANIFEST dist ${package}.egg-info
	-${RM} -r `find . -name '__pycache__'`
	-${RM} `find . -name '*.py[co]'`
fullclean: distclean srcclean docsclean
	-find . -name '*~' -exec rm -f {} ';'

# ----

.PHONY: install uninstall
install: build
	${PYTHON} setup.py install --prefix='' --user ${INSTALLOPT}
uninstall:
	-${RM} -r $(shell ${PYTHON} -m site --user-site)/${package}
	-${RM} -r $(shell ${PYTHON} -m site --user-site)/${package}-*-py*.egg-info

# ----

.PHONY: docs docs-html docs-pdf docs-misc
docs: docs-html docs-pdf docs-misc
docs-html: rst2html sphinx-html epydoc-html
docs-pdf:  sphinx-pdf epydoc-pdf
docs-misc: sphinx-man sphinx-info

RST2HTML = $(shell command -v rst2html || command -v rst2html.py || false)
RST2HTMLOPTS  = --input-encoding=utf-8
RST2HTMLOPTS += --no-compact-lists
RST2HTMLOPTS += --cloak-email-addresses
.PHONY: rst2html
rst2html:
	${RST2HTML} ${RST2HTMLOPTS} ./LICENSE.rst  > docs/LICENSE.html
	${RST2HTML} ${RST2HTMLOPTS} ./CHANGES.rst  > docs/CHANGES.html
	${RST2HTML} ${RST2HTMLOPTS} docs/index.rst > docs/index.html

SPHINXBUILD = sphinx-build
SPHINXOPTS  =
.PHONY: sphinx sphinx-html sphinx-pdf sphinx-man sphinx-info
sphinx: sphinx-html sphinx-pdf sphinx-man sphinx-info
sphinx-html:
	${PYTHON} -c 'import ${package}.${MODULE}'
	mkdir -p build/doctrees docs/usrman
	${SPHINXBUILD} -b html -d build/doctrees ${SPHINXOPTS} \
	docs/source docs/usrman
	${RM} docs/usrman/.buildinfo
sphinx-pdf:
	${PYTHON} -c 'import ${package}.${MODULE}'
	mkdir -p build/doctrees build/latex
	${SPHINXBUILD} -b latex -d build/doctrees ${SPHINXOPTS} \
	docs/source build/latex
	${MAKE} -C build/latex all-pdf > /dev/null
	mv build/latex/*.pdf docs/
sphinx-man:
	${PYTHON} -c 'import ${package}.${MODULE}'
	mkdir -p build/doctrees build/man
	${SPHINXBUILD} -b man -d build/doctrees ${SPHINXOPTS} \
	docs/source build/man
	mv build/man/*.[137] docs/
sphinx-info:
	${PYTHON} -c 'import ${package}.${MODULE}'
	mkdir -p build/doctrees build/texinfo
	${SPHINXBUILD} -b texinfo -d build/doctrees ${SPHINXOPTS} \
	docs/source build/texinfo
	${MAKE} -C build/texinfo info > /dev/null
	mv build/texinfo/*.info docs/

EPYDOCBUILD = ${PYTHON} ./conf/epydocify.py
EPYDOCOPTS  =
.PHONY: epydoc epydoc-html epydoc-pdf
epydoc: epydoc-html epydoc-pdf
epydoc-html:
	${PYTHON} -c 'import ${package}.${MODULE}'
	mkdir -p docs/apiref
	${EPYDOCBUILD} ${EPYDOCOPTS} --html -o docs/apiref
epydoc-pdf:

.PHONY: docsclean
docsclean:
	-${RM} docs/*.info docs/*.[137]
	-${RM} docs/*.html docs/*.pdf
	-${RM} -r docs/usrman docs/apiref

# ----

.PHONY: sdist
sdist: srcbuild docs
	${PYTHON} setup.py sdist ${SDISTOPT}

# ----
